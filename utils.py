
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.functional import resize
from torchvision.transforms import functional as F
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image
import base64
import io
import os
from typing import Union
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from torchvision import datasets
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from sklearn.manifold import TSNE
from sklearn.cluster import k_means
from sklearn.decomposition import PCA

from EfficentRex import EfficentRex
from RexNet import RexNet

def visualize_image(image_tensor):
    """
    Visualize a PyTorch image tensor
    
    Args:
    - image_tensor: A torch tensor of shape [3, 224, 224] or [1, 3, 224, 224]
    """
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)
    
    image_np = image_tensor.cpu().numpy()

    image_np = np.transpose(image_np, (1, 2, 0))
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image_np = image_np * std + mean
    
    # Clip values to 0-1 range
    image_np = np.clip(image_np, 0, 1)
    
    # Plot the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image_np)
    plt.axis('off')
    plt.show()

def letterbox_to_square(img, size=256, fill=0):
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = int(round(w * scale * 0.8)), int(round(h * scale * 0.8))
    img = F.resize(img, (new_h, new_w), antialias=True)
    
    pad_left   = (size - new_w) // 2
    pad_right  = size - new_w - pad_left
    pad_top    = (size - new_h) // 2
    pad_bottom = size - new_h - pad_top
    img = F.pad(img, [pad_left, pad_top, pad_right, pad_bottom], fill=fill)
    return img

# ---------    --------- #

def embed_test_given_ckpt_path(
    ckpt_path: str = "log/lightning_logs/version_1/checkpoints/best-epoch=39-val_loss=0.0433.ckpt", 
    data_dir: str = "dataset/dataset",
    _model: str = 'EfficentRex',
    device: str = "cpu",
    return_dataset: bool = False):
    
    config = OmegaConf.load(f"config/config_{_model.lower()}.yaml")
       
    if _model.lower() == "rexnet":
        model_class = RexNet
    elif _model.lower() == "efficentrex":
        model_class = EfficentRex
    
    model = model_class.load_from_checkpoint(
        ckpt_path,
        config=config.model,
        strict=False
    )
    model.eval()
    model.to(device)
    
    # load dataset
    val_tfms = transforms.Compose([
        transforms.Lambda(lambda im: im.convert("RGB")),
        transforms.Lambda(lambda im: letterbox_to_square(im, size=config.size, fill=0)),
        transforms.CenterCrop(224),
        model.base_tfms,
    ])

    full_test = datasets.ImageFolder(root=f"{data_dir}/train", transform=val_tfms)  # Same data, different transforms



    # ---- take 10% of the dataset ----
    # percent = 0.05
    # num_samples = int(len(full_test) * percent)

    # # reproducible random selection
    # np.random.seed(42)
    # subset_indices = np.random.choice(len(full_test), num_samples, replace=False)

    # small_dataset = Subset(full_test, subset_indices)

    # loader = DataLoader(
    #     small_dataset,
    #     batch_size=config.training.batch_size,
    #     num_workers=config.training.num_workers
    # )
    
    idx_to_class = {idx: class_ for class_, idx in full_test.class_to_idx.items()}
    
    loader = DataLoader(
        full_test, 
        batch_size=config.training.batch_size, 
        num_workers=config.training.num_workers)

    # embedd images
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            data, labels = batch
            embeddings = model.get_latent_rapresentation(data)
            if idx == 1:
                print(embeddings.shape)
            all_embeddings.append(embeddings)
            all_labels.append(labels)

    all_embeddings = torch.concat(all_embeddings)
    all_labels = torch.concat(all_labels)
    
    accuracy, results = get_test_accuracy(
        model, 
        loader,
        idx_to_class,
        device)
    
    if not return_dataset:
        return all_embeddings, all_labels, accuracy
    else:
        return all_embeddings, all_labels, accuracy, full_test

def get_test_accuracy(
    model, 
    loader,
    idx_to_class,
    device):

    results = {
            "scores": [],
            "predicted_names": [],
            "real_names": [],
            "correct": [],
            "overall_accuracy": None,
        }

    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x).softmax(1)
            predicted_class_id = pred.argmax(1)

            idx = torch.arange(len(predicted_class_id))
            score = pred[idx, predicted_class_id]
            
            correct = (predicted_class_id == y).float()

            pred_cpu = predicted_class_id.cpu()
            y_cpu = y.cpu()
            score_cpu = score.cpu()
            correct_cpu = correct.cpu()

            results["scores"].extend(score_cpu.tolist())
            results["predicted_names"].extend([idx_to_class[int(c)] for c in pred_cpu])
            results["real_names"].extend([idx_to_class[int(c)] for c in y_cpu])
            results["correct"].extend(correct_cpu.tolist())

            total_correct += int(correct_cpu.sum().item())
            total_samples += len(y_cpu)
            
    results["overall_accuracy"] = total_correct / total_samples
    
    return results["overall_accuracy"], results

def tsne_and_cluster_from_ckpt(    
    ckpt_path: str = "log/lightning_logs/version_1/checkpoints/best-epoch=39-val_loss=0.0433.ckpt",
    data_dir: str = "dataset/dataset", 
    _model: str = 'EfficentRex',
    device: str = "cpu"):
    
    embeddings, labels, test_accuracy = embed_test_given_ckpt_path(ckpt_path, data_dir, _model, device)
    
    print(embeddings.shape)
    
    if embeddings.shape[1] > 2:
        tsne = TSNE(
            n_components=2, 
            learning_rate='auto'
            ).fit_transform(embeddings)
    else:
        tsne = embeddings

    centroids, clusters, inertia = k_means(
        tsne, 
        n_clusters=8,
        n_init=5)
    
    k_means_acc, y_pred = cluster_majority_vote_accuracy(labels, clusters)    
    
    return embeddings, labels, tsne, clusters, k_means_acc, test_accuracy

def cluster_majority_vote_accuracy(labels, clusters):
    def _to_numpy(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    y = _to_numpy(labels).ravel()
    c = _to_numpy(clusters).ravel()

    mapping = {}
    for cid in np.unique(c):
        mask = c == cid
        if not np.any(mask):
            continue
        vals, counts = np.unique(y[mask], return_counts=True)
        mapping[cid] = vals[np.argmax(counts)]

    y_pred = np.array([mapping.get(cid, -1) for cid in c])
    acc = float((y_pred == y).mean())
    return acc, y_pred

## Plotting functions

def plot_tsne(
    embeddings: Union[np.ndarray, torch.Tensor], 
    labels: Union[np.ndarray, torch.Tensor]):
    
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap="tab10", s=5, alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Classes", loc="best", bbox_to_anchor=(1,1))
    plt.title("t-SNE")
    plt.show()

def interactive_tsne_over_checkpoints(
    ckpt_dir: str,
    ckpt_files: list[str],
    _model: str = "EfficentRex",
    data_dir: str = "dataset/dataset",
    point_size: int = 25,
    save_html: str | None = None,
):

    import numpy as np
    import pandas as pd
    import altair as alt
    
    dfs = []
    accuracies = {}
    kmeans_accuracies = {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
    
    for i, ckpt_file in enumerate(ckpt_files):
        print(f"Checkpoint: {ckpt_file}")
        ckpt_path = os.path.join(ckpt_dir, ckpt_file)
        # extract embeddigs, cluster and tsne
        embeddings, labels, tsne, clusters, k_means_acc, test_acc = tsne_and_cluster_from_ckpt(
            ckpt_path=ckpt_path,
            data_dir=data_dir,
            _model=_model,
            device=device,
        )
        
        kmeans_accuracies[i] = k_means_acc 
        accuracies[i] = test_acc

        X = np.asarray(tsne)
        c = np.asarray(clusters).astype(int)
        y = np.asarray(labels).astype(int)

        df_i = pd.DataFrame({
            "x": X[:, 0],
            "y": X[:, 1],
            "cluster": c.astype(str),
            "label": y.astype(str),
            "ckpt_idx": i,
            "ckpt": ckpt_file,
            "test_accuracy": test_acc,
        })
        dfs.append(df_i)

    df_all = pd.concat(dfs, ignore_index=True)

    alt.data_transformers.disable_max_rows()

    # color toggle and checkpoint slider
    color_toggle = alt.param(
        name="color_by",
        value="cluster",
        bind=alt.binding_radio(options=["cluster", "label"], name="Color by: "),
    )

    ckpt_slider = alt.param(
        name="ckpt_idx",
        value=0,
        bind=alt.binding_range(min=0, max=len(ckpt_files) - 1, step=1, name="Checkpoint: "),
    )

    points = (
        alt.Chart(df_all)
        .transform_calculate(
            color="color_by == 'cluster' ? datum.cluster : datum.label"
        )
        .transform_filter("datum.ckpt_idx == ckpt_idx")
        .mark_point(filled=True, opacity=0.75)
        .encode(
            x=alt.X("x:Q", axis=alt.Axis(title="t-SNE 1")),
            y=alt.Y("y:Q", axis=alt.Axis(title="t-SNE 2")),
            color=alt.Color("color:N", legend=alt.Legend(title="Color")),
            tooltip=[
                # alt.Tooltip("ckpt:N", title="Checkpoint"),
                # alt.Tooltip("test_accuracy:Q", title="Test Accuracy", format=".2f"),
                alt.Tooltip("cluster:N", title="Cluster"),
                alt.Tooltip("label:N", title="Label"),
                # alt.Tooltip("x:Q", format=".2f"),
                # alt.Tooltip("y:Q", format=".2f"),
            ],
            size=alt.value(point_size),
        )
        .add_params(color_toggle, ckpt_slider)
    )
    
    # small DataFrame for the accuracy
    acc_df = pd.DataFrame([
        {
            "ckpt_idx": i,
            "ckpt": ckpt_files[i],
            "test_text": f"Test acc: {accuracies[i]:.2f}",
            "kmeans_text": f"k-means acc: {kmeans_accuracies[i]:.2f}",
        }
        for i in accuracies.keys()
    ])
    
    accuracy_text = (
        alt.Chart(acc_df)
        .transform_filter("datum.ckpt_idx == ckpt_idx")
        .mark_text(align="right", baseline="top", dx=-10, dy=10, fontSize=14, fontWeight="bold", color="black")
        .encode(
            x=alt.value(690),
            y=alt.value(10),
            text=alt.Text("test_text:N"),
        )
    )

    kmeans_text = (
        alt.Chart(acc_df)
        .transform_filter("datum.ckpt_idx == ckpt_idx")
        .mark_text(align="right", baseline="top", dx=-10, dy=28, fontSize=14, fontWeight="bold", color="black")
        .encode(
            x=alt.value(690),
            y=alt.value(28),
            text=alt.Text("kmeans_text:N"),
        )
    )

    chart = points + accuracy_text + kmeans_text

    chart = chart.properties(
        width=700,
        height=600,
        title=f"t-SNE across checkpoints ({_model}) - Val embeddings, Test accuracy shown",
    ).interactive()

    if save_html:
        chart.save(save_html)

    return chart

def interactive_tsne_with_images(
    ckpt_path: str,
    _model: str = "cnn",
    config_path: str = "config.yaml",
    data_dir: str = "dataset/dataset",
    n_samples: int = 500,
    image_size: int = 14,
    point_size: int = 25,
    show_centroids: bool = True,
    save_html: str | None = None,
    random_seed: int = 42,
    overlay_scale: int = 8,
):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
       
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
   
    embeddings, labels, test_accuracy, test_dataset = embed_test_given_ckpt_path(
        ckpt_path, data_dir, _model, device, return_dataset=True)

    # subsample
    N = len(embeddings)
    if N > n_samples:
        sel = np.random.choice(N, n_samples, replace=False)
        embeddings = embeddings[sel]
        labels = labels[sel]
        subsampled_idx = sel
    else:
        subsampled_idx = np.arange(N)

    imgs = []
    for i in subsampled_idx:
        img_tensor, _ = test_dataset[i]
        imgs.append(img_tensor)
    imgs = torch.stack(imgs, dim=0)

    def to_data_url(img_tensor, target_size=14):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img = img_tensor.clone() * std + mean
        img = torch.clamp(img, 0, 1)
        arr = (img.squeeze().numpy() * 255).astype(np.uint8)
        pil = Image.fromarray(arr, mode="L").resize((target_size, target_size), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    image_urls = [to_data_url(t, image_size) for t in imgs]

    E = embeddings.detach().cpu().numpy() if torch.is_tensor(embeddings) else np.asarray(embeddings)
    if E.ndim == 2 and E.shape[1] > 2:
        tsne = TSNE(n_components=2, learning_rate="auto", random_state=random_seed).fit_transform(E)
    else:
        tsne = E

    _, clusters, _ = k_means(tsne, n_clusters=10, n_init=5, random_state=random_seed)

    y = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
    df = pd.DataFrame({
        "x": tsne[:, 0],
        "y": tsne[:, 1],
        "cluster": clusters.astype(str),
        "label": y.astype(str),
        "image_url": image_urls,
        "row_id": np.arange(len(tsne)),
    })

    test_acc = get_test_accuracy(ckpt_path, _model, config_path)

    alt.data_transformers.disable_max_rows()

    tsne_w, tsne_h = 700, 600
    panel_pad = 8
    panel_img_w = int(image_size * overlay_scale)
    panel_img_h = int(image_size * overlay_scale)
    panel_w = panel_img_w + 2 * panel_pad
    panel_h = panel_img_h + 2 * panel_pad

    # toggle color by cluster or label
    color_toggle = alt.param(
        name="color_by",
        value="cluster",
        bind=alt.binding_radio(options=["cluster", "label"], name="Color by: "),
    )

    hover_sel = alt.selection_point(
        name="hover",
        on="mouseover",
        fields=["row_id"],
        nearest=True,
        empty="none",
    )

    # tsne points panel
    points = (
        alt.Chart(df)
        .mark_point(filled=True, opacity=0.75)
        .encode(
            x=alt.X("x:Q", axis=alt.Axis(title="t-SNE 1")),
            y=alt.Y("y:Q", axis=alt.Axis(title="t-SNE 2")),
            color=alt.Color("color:N", legend=alt.Legend(title="Color")),
            tooltip=[
                alt.Tooltip("label:N", title="Label"),
                alt.Tooltip("cluster:N", title="Cluster"),
            ],
        )
        .transform_calculate(
            color="color_by == 'cluster' ? datum.cluster : datum.label"
        )
        .add_params(color_toggle, hover_sel)
        .properties(width=tsne_w, height=tsne_h, title="t-SNE (val)")
    )

    # optional centroids
    layers_tsne = [points]
    if show_centroids:
        cent = (
            pd.DataFrame({"x": tsne[:, 0], "y": tsne[:, 1], "cluster": clusters})
            .groupby("cluster")
            .mean()
            .reset_index()
            .rename(columns={"x": "cx", "y": "cy"})
        )
        cent["cluster"] = cent["cluster"].astype(str)

        centroid_layer = (
            alt.Chart(cent)
            .transform_calculate(
                centroid_opacity="color_by == 'cluster' ? 1 : 0"
            )
            .mark_point(shape="cross", size=220, filled=False, stroke="black", strokeWidth=1.5)
            .encode(
                x="cx:Q",
                y="cy:Q",
                opacity=alt.Opacity("centroid_opacity:Q", legend=None),
                tooltip=[alt.Tooltip("cluster:N", title="Centroid (cluster)")],
            )
            .add_params(color_toggle)
        )
        layers_tsne.append(centroid_layer)

    # test accuracy
    accuracy_text = (
        alt.Chart(pd.DataFrame([{"test_accuracy": test_acc}]))
        .mark_text(
            align="right",
            baseline="top",
            dx=-10,
            dy=10,
            fontSize=14,
            fontWeight="bold",
            color="black",
        )
        .encode(
            x=alt.value(tsne_w),
            y=alt.value(0),
            text=alt.Text("test_accuracy:Q", format=".2f"),
        )
    )
    layers_tsne.append(accuracy_text)

    tsne_panel = alt.layer(*layers_tsne).properties(width=tsne_w, height=tsne_h).interactive()

    pos_x = panel_w / 2
    pos_y = panel_h / 2

    # separate image panel
    image_panel = (
        alt.Chart(df)
        .transform_filter(hover_sel)
        .mark_image(width=panel_img_w, height=panel_img_h)
        .encode(
            url=alt.Url("image_url:N"),
            x=alt.value(pos_x),
            y=alt.value(pos_y),
        )
        .add_params(hover_sel)
        .properties(width=panel_w, height=panel_h, title="Image")
    )

    chart = alt.hconcat(tsne_panel, image_panel).resolve_scale(color="independent")

    if save_html:
        chart.save(save_html)

    return chart
