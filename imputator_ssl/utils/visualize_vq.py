import os
# os.environ['OPENBLAS_NUM_THREADS'] = '1'  # 限制 OpenBLAS 线程数
# print(os.environ.get('OPENBLAS_NUM_THREADS'))
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_vq_embedding_space_continuous(embeddings, labels, label_dim=0, perplexity=30):
    """
    使用 t-SNE 可视化 VQ-VAE 量化嵌入向量的序列级别分布（2D 空间，带连续标签）。

    参数：
        embeddings: 量化嵌入向量，形状 (num, embedding_dim, num_vectors)，例如 (num, 64, 12)。
        labels: 连续标签值，形状 (num,) 或 (num, labels)，每个序列对应一个或多个标签。
        label_dim: 如果 labels 是 (num, labels)，选择要使用的标签维度（默认为第一个维度）。
        perplexity: t-SNE 的困惑度参数，控制局部结构的关注程度。
    """
    # 确保输入为 numpy 数组
    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels)
    # 生成随机索引并打乱
    indices = np.random.permutation(len(embeddings))

    # 取前2000个打乱后的索引
    indices = indices[:2000]

    # 根据打乱的索引重新排序 embeddings 和 labels
    embeddings = embeddings[indices]
    labels = labels[indices]

    # 检查输入形状
    if len(embeddings.shape) != 3:
        raise ValueError(f"期望 embeddings 形状为 (num, embedding_dim, num_vectors)，实际为 {embeddings.shape}")
    num, embedding_dim, num_vectors = embeddings.shape

    # 处理标签形状
    if len(labels.shape) == 1:
        labels = labels
    elif len(labels.shape) == 2:
        if labels.shape[0] != num:
            raise ValueError(f"标签数量 ({labels.shape[0]}) 必须匹配序列数量 ({num})")
        labels = labels[:, label_dim]
    else:
        raise ValueError(f"期望 labels 形状为 (num,) 或 (num, labels)，实际为 {labels.shape}")

    # 展平嵌入向量
    embeddings_flat = embeddings.transpose(0, 2, 1).reshape(num, embedding_dim * num_vectors)

    # 使用 t-SNE 降维到 2D
    reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings_flat)

    # 绘制 2D 散点图，带连续标签的颜色梯度
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
    plt.colorbar(scatter).set_label('Mean NDVI of Time Series', fontsize=14)
    plt.title('Visualization of Learned Latent Space (t-SNE)', fontsize=16)
    plt.grid(True)

    # 保存图像
    plt.savefig(f'vq_embedding_space_tsne_sequence_dim{label_dim}.png')
    plt.close()


def visualize_vq_embedding_space_categorical(embeddings, labels, perplexity=30):
    """
    使用 t-SNE 可视化 VQ-VAE 量化嵌入向量的序列级别分布（2D 空间，带分类标签）。

    参数：
        embeddings: 量化嵌入向量，形状 (num, embedding_dim, num_vectors)，例如 (num, 64, 12)。
        labels: 分类标签（字符串），形状 (num,)，例如 ['Water', 'developed', ...]。
        perplexity: t-SNE 的困惑度参数，控制局部结构的关注程度。
    """
    # 确保输入为 numpy 数组
    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels)

    # 检查输入形状
    if len(embeddings.shape) != 3:
        raise ValueError(f"期望 embeddings 形状为 (num, embedding_dim, num_vectors)，实际为 {embeddings.shape}")
    num, embedding_dim, num_vectors = embeddings.shape

    if len(labels.shape) != 1:
        raise ValueError(f"期望 labels 形状为 (num,)，实际为 {labels.shape}")
    if labels.shape[0] != num:
        raise ValueError(f"标签数量 ({labels.shape[0]}) 必须匹配序列数量 ({num})")

    # 将字符串标签转换为整数编码
    unique_labels = np.unique(labels)
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    labels_int = np.array([label_to_int[label] for label in labels])

    # 展平嵌入向量
    embeddings_flat = embeddings.transpose(0, 2, 1).reshape(num, embedding_dim * num_vectors)

    # 使用 t-SNE 降维到 2D
    reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings_flat)

    # 绘制 2D 散点图，使用离散颜色
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=labels_int,
        cmap='tab10',  # 使用离散颜色映射
        s=50,
        alpha=0.7
    )

    # 添加图例
    handles, _ = scatter.legend_elements(prop="colors")
    plt.legend(
        handles,
        unique_labels,
        title="LCMAP Category",
        loc="best",
        fontsize=12
    )

    # 设置标题和网格
    plt.title('Visualization of Learned Latent Space (t-SNE) with LCMAP Categories', fontsize=16)
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)
    plt.grid(True)

    # 保存图像
    plt.savefig('vq_embedding_space_tsne_categorical.png')
    plt.close()

    print("分类标签可视化已保存为 'vq_embedding_space_tsne_categorical.png'")