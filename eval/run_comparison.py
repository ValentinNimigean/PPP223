import subprocess
import json
import os
import sys
import time
import statistics

def run_eval(chunker_type, output_path, demo_mode=False):
    print(f"\n==================================================")
    print(f"Running evaluation for chunker: {chunker_type.upper()} (Demo Mode: {demo_mode})")
    print(f"==================================================")
    
    cmd = [
        sys.executable,
        "eval/eval.py",
        "--repo", ".",
        "--chunker", chunker_type,
        "--out", output_path,
    ]
    if not demo_mode:
        cmd.append("--disable-deterministic-shortcuts")
    
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.perf_counter() - t0
    
    if result.returncode != 0:
        print(f"Error: Evaluation failed for chunker {chunker_type}")
        return None
        
    print(f"Completed {chunker_type} evaluation in {elapsed:.2f} seconds.")
    return output_path

def compile_results(iterations=1):
    chunkers = ["ast", "line", "char"]
    results = {}
    
    for chunker in chunkers:
        ingest_times = []
        retrieval_latencies = []
        entity_scores = []
        file_scores = []
        combined_scores = []
        penalized_combined_scores = []
        hallucination_rates = []
        
        runs_counted = 0
        for i in range(iterations):
            report_path = f"evaluation_reports/eval_report_{chunker}_run_{i}.json" if iterations > 1 else f"evaluation_reports/eval_report_{chunker}.json"
            if not os.path.exists(report_path):
                continue
                
            with open(report_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            overall = data.get("overall", {})
            results_list = data.get("results", [])
            
            # Calculate hallucination rate for this run
            hallucination_queries = sum(1 for r in results_list if r.get("hallucination_penalty", 0.0) > 0.0)
            hallucination_rate = (hallucination_queries / len(results_list)) if results_list else 0.0
            
            ingest_times.append(data.get("ingest_time_seconds", 0.0))
            retrieval_latencies.append(data.get("average_retrieval_latency_seconds", 0.0))
            entity_scores.append(overall.get("entity_score", 0.0))
            file_scores.append(overall.get("file_score", 0.0))
            combined_scores.append(overall.get("combined_score", 0.0))
            penalized_combined_scores.append(overall.get("penalized_combined_score", 0.0))
            hallucination_rates.append(hallucination_rate)
            runs_counted += 1
            
        if runs_counted == 0:
            print(f"Warning: No reports found for chunker {chunker}.")
            continue
            
        results[chunker] = {
            "ingest_time_seconds": statistics.mean(ingest_times),
            "average_retrieval_latency_seconds": statistics.mean(retrieval_latencies),
            "file_score": statistics.mean(file_scores),
            "entity_score": statistics.mean(entity_scores),
            "hallucination_rate": statistics.mean(hallucination_rates),
            "combined_score": statistics.mean(combined_scores),
            "penalized_combined_score": statistics.mean(penalized_combined_scores),
            "runs_evaluated": runs_counted
        }
        
    # Print comparison table in markdown format
    print(f"\n\n### Evaluation Results Comparison (Average of {iterations} runs)\n")
    
    md_table = []
    md_table.append("| Metric | AST-Based Chunker (Ours) | Line-Based Chunker | Character-Based Chunker |")
    md_table.append("| :--- | :---: | :---: | :---: |")
    
    metrics_mapping = [
        ("Ingestion / Chunking Time", "ingest_time_seconds", "{:.3f} s"),
        ("Avg. Retrieval Latency", "average_retrieval_latency_seconds", "{:.4f} s"),
        ("Retrieval Recall (File Score)", "file_score", "{:.2%}"),
        ("Entity Match Rate", "entity_score", "{:.2%}"),
        ("Hallucination Rate", "hallucination_rate", "{:.2%}"),
        ("Overall Combined Score", "combined_score", "{:.2f}"),
        ("Penalized Combined Score", "penalized_combined_score", "{:.2f}"),
    ]
    
    for metric_name, key, fmt in metrics_mapping:
        row = f"| {metric_name} "
        for chunker in chunkers:
            val = results.get(chunker, {}).get(key, 0.0)
            row += f"| {fmt.format(val)} "
        row += "|"
        md_table.append(row)
        
    for line in md_table:
        print(line)
        
    # Also generate LaTeX table format
    latex_table = []
    latex_table.append(r"\begin{table}[h]")
    latex_table.append(r"\centering")
    latex_table.append(r"\begin{tabular}{lccc}")
    latex_table.append(r"\hline")
    latex_table.append(r"Metric & AST-Based & Line-Based & Character-Based \\")
    latex_table.append(r"\hline")
    
    for metric_name, key, fmt in metrics_mapping:
        row = f"{metric_name} "
        for chunker in chunkers:
            val = results.get(chunker, {}).get(key, 0.0)
            row += f"& {fmt.format(val)} "
        row += r"\\"
        latex_table.append(row)
        
    latex_table.append(r"\hline")
    latex_table.append(r"\end{tabular}")
    latex_table.append(r"\caption{Comparison of Chunking Strategies (Averaged over " + str(iterations) + r" Runs)}")
    latex_table.append(r"\label{tab:chunking_comparison}")
    latex_table.append(r"\end{table}")
    
    print("\n\n### LaTeX Table Code\n")
    for line in latex_table:
        print(line)
        
    # Save the reports
    with open("evaluation_reports/comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    with open("evaluation_reports/comparison_table.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_table))
        
    with open("evaluation_reports/comparison_table.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(latex_table))

if __name__ == "__main__":
    demo_flag = "--demo" in sys.argv
    
    # Parse iterations
    iterations = 1
    for idx, arg in enumerate(sys.argv):
        if arg == "--iterations" and idx + 1 < len(sys.argv):
            try:
                iterations = int(sys.argv[idx + 1])
            except ValueError:
                pass

    if len(sys.argv) > 1 and sys.argv[1] == "--compile-only":
        compile_results(iterations=iterations)
    else:
        chunkers = ["ast", "line", "char"]
        for chunker in chunkers:
            for i in range(iterations):
                report_path = f"evaluation_reports/eval_report_{chunker}_run_{i}.json" if iterations > 1 else f"evaluation_reports/eval_report_{chunker}.json"
                print(f"\n--- Starting Chunker: {chunker.upper()} | Run {i+1}/{iterations} ---")
                run_eval(chunker, report_path, demo_mode=demo_flag)
                
        compile_results(iterations=iterations)
