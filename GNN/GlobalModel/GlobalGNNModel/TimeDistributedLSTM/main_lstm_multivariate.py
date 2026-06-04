# =============================================================================
# Main
# =============================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Loading data from {DATA_PATH}...")

    df = pd.read_feather(DATA_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([DATE_COL, 'item_id']).reset_index(drop=True)

    clusters, df_wide, cat_labels = build_similarity_clusters(df)
    if not clusters:
        print("No clusters of sufficient size found. Nothing to train.")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_rows = []
    overall_start = time.time()
    for cluster_idx, item_ids in enumerate(clusters):
        try:
            all_rows.extend(run_cluster(cluster_idx, item_ids, df_wide, cat_labels, device))
        except Exception as exc:                          # keep going on failure
            import traceback
            print(f"[ERROR] cluster {cluster_idx}: {exc}")
            traceback.print_exc()

    if all_rows:
        results = pd.DataFrame(all_rows)
        per_series_path = os.path.join(RESULTS_DIR, 'multivariate_per_series_results.csv')
        results.to_csv(per_series_path, index=False)

        summary = (
            results.groupby(['cluster', 'cluster_size'])
            .agg(mean_rmse=('rmse', 'mean'), mean_mae=('mae', 'mean'),
                 mean_bias=('bias', 'mean'), mean_pocid=('pocid', 'mean'),
                 mean_score=('score', 'mean'))
            .reset_index()
        )
        summary_path = os.path.join(RESULTS_DIR, 'multivariate_cluster_summary.csv')
        summary.to_csv(summary_path, index=False)

        print(f"\nDone in {(time.time() - overall_start) / 60:.1f} min.")
        print(f"  Per-series results : {per_series_path}")
        print(f"  Cluster summary    : {summary_path}")
        print(f"  Overall mean RMSE  : {results['rmse'].mean():.4f} "
              f"over {len(results)} series in {results['cluster'].nunique()} clusters")
    else:
        print("No results produced.")


if __name__ == '__main__':
    main()
