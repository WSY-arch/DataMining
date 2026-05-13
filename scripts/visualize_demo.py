"""
完整的可视化演示脚本
"""

import argparse
import numpy as np
from pathlib import Path

from tsclust.clustering import cluster_time_series
from tsclust.visualization import (
    plot_clustering_results,
    plot_distance_matrix,
    plot_medoids_comparison,
    plot_cluster_statistics,
)


def generate_demo_data(n_per_cluster: int, length: int, random_state: int) -> np.ndarray:
    """生成演示数据"""
    rng = np.random.default_rng(random_state)
    t = np.linspace(0, 2 * np.pi, length)
    data = []
    
    # 类别1: sin(t) + 噪声
    for _ in range(n_per_cluster):
        data.append(np.sin(t) + 0.1 * rng.standard_normal(length))
    
    # 类别2: cos(t) + 噪声
    for _ in range(n_per_cluster):
        data.append(np.cos(t) + 0.1 * rng.standard_normal(length))
    
    # 类别3: sin(2t) + 噪声
    for _ in range(n_per_cluster):
        data.append(np.sin(2 * t) + 0.1 * rng.standard_normal(length))
    
    return np.vstack(data)


def load_csv(path: str) -> np.ndarray:
    """从CSV加载数据"""
    return np.loadtxt(path, delimiter=",")


def main():
    parser = argparse.ArgumentParser(
        description="Isolation Kernel + k-medoids 时间序列聚类（带可视化）"
    )
    parser.add_argument("--input", type=str, default="", help="CSV数据路径（每行为一个序列）")
    parser.add_argument("--k", type=int, default=3, help="聚类数量")
    parser.add_argument("--n-trees", type=int, default=100, help="isolation树数量")
    parser.add_argument("--sample-size", type=int, default=256, help="子样本大小")
    parser.add_argument("--length", type=int, default=100, help="演示数据序列长度")
    parser.add_argument("--per-cluster", type=int, default=20, help="演示数据每簇样本数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output-dir", type=str, default="", help="输出图表目录（不指定则只显示）")
    parser.add_argument("--no-plot", action="store_true", help="不显示交互式图表")
    
    args = parser.parse_args()
    
    # 加载数据
    if args.input:
        print(f"📁 从 {args.input} 加载数据...")
        X = load_csv(args.input)
    else:
        print(f"🔄 生成演示数据: {args.per_cluster} 个样本/簇 × {args.k} 簇, 序列长度 {args.length}")
        X = generate_demo_data(args.per_cluster, args.length, args.seed)
    
    print(f"📊 数据形状: {X.shape}\n")
    
    # 执行聚类
    print(f"🚀 执行聚类...")
    print(f"   - k-medoids 簇数: {args.k}")
    print(f"   - Isolation树数: {args.n_trees}")
    print(f"   - 子样本大小: {args.sample_size}")
    print(f"   - 随机种子: {args.seed}\n")
    
    result = cluster_time_series(
        X,
        k=args.k,
        n_trees=args.n_trees,
        sample_size=args.sample_size,
        normalize=True,
        random_state=args.seed,
    )
    
    # 打印结果统计
    print("✅ 聚类完成！\n")
    print(f"📈 结果统计:")
    print(f"   - Medoid索引: {result.medoids}")
    print(f"   - 样本标签（前20个）: {result.labels[:20]}")
    
    for cluster_id in range(args.k):
        n_samples = np.sum(result.labels == cluster_id)
        print(f"   - 簇 {cluster_id}: {n_samples} 个样本")
    
    print()
    
    # 计算聚类质量指标
    print("📊 聚类质量:")
    for cluster_id in range(args.k):
        cluster_mask = result.labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        if len(cluster_indices) > 1:
            cluster_dist = result.distance_matrix[np.ix_(cluster_indices, cluster_indices)]
            mask = ~np.eye(len(cluster_indices), dtype=bool)
            avg_intra = np.mean(cluster_dist[mask])
            print(f"   - 簇 {cluster_id} 平均簇内距离: {avg_intra:.4f}")
    
    print()
    
    # 生成输出目录路径
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 将保存图表到: {output_dir}\n")
    else:
        output_dir = None
    
    # 生成可视化
    print("🎨 生成可视化...\n")
    
    # 1. 聚类结果（各聚类中的时间序列）
    print("   [1/4] 绘制聚类结果...")
    plot_path_1 = str(output_dir / "1_clustering_results.png") if output_dir else None
    if not args.no_plot or output_dir:
        plot_clustering_results(
            X,
            result.labels,
            result.medoids,
            title="时间序列聚类结果（按聚类分组）",
            figsize=(14, 3 * args.k),
            save_path=plot_path_1,
        )
    
    # 2. 距离矩阵热力图
    print("   [2/4] 绘制距离矩阵热力图...")
    plot_path_2 = str(output_dir / "2_distance_matrix.png") if output_dir else None
    if not args.no_plot or output_dir:
        plot_distance_matrix(
            result.distance_matrix,
            labels=result.labels,
            title="距离矩阵（按聚类重排序）",
            figsize=(10, 9),
            save_path=plot_path_2,
        )
    
    # 3. Medoid对比
    print("   [3/4] 绘制Medoid对比...")
    plot_path_3 = str(output_dir / "3_medoids_comparison.png") if output_dir else None
    if not args.no_plot or output_dir:
        plot_medoids_comparison(
            X,
            result.labels,
            result.medoids,
            title="各聚类的Medoid序列",
            figsize=(4 * args.k, 4),
            save_path=plot_path_3,
        )
    
    # 4. 聚类统计
    print("   [4/4] 绘制聚类统计...")
    plot_path_4 = str(output_dir / "4_cluster_statistics.png") if output_dir else None
    if not args.no_plot or output_dir:
        plot_cluster_statistics(
            result.distance_matrix,
            result.labels,
            title="聚类质量统计",
            figsize=(12, 4),
            save_path=plot_path_4,
        )
    
    print("\n✅ 所有可视化完成！")
    
    if output_dir:
        print(f"\n📸 生成的图表:")
        print(f"   1️⃣  {plot_path_1}")
        print(f"   2️⃣  {plot_path_2}")
        print(f"   3️⃣  {plot_path_3}")
        print(f"   4️⃣  {plot_path_4}")


if __name__ == "__main__":
    main()
