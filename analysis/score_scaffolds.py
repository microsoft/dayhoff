import pandas as pd 
import os 

models = ['dayhoff-3b', 'dayhoff-170m', 'evodiff', 'jamba-170m-10mbothfilter-36w/', 'jamba-170m-10mnofilter-36w/', 'jamba-170m-gigaclust-36w/', 'jamba-170m-seq-36w/', 'jamba-170m-seqsam-36w/', 'jamba-3b-indel-gigaclust-120k-2/', 'jamba-3b-seq-sam-biar-fsdp-tok90k/']
problem_set = ['motifbench', 'rfdiff']

mega_df = []
alpha = 5

for problem in problem_set:
    for model in models: 
        result_file = f'model-evals/scaffolding-results/{problem}/{model}/foldseek_results.csv'
        if os.path.exists(result_file):
            df = pd.read_csv(result_file)
            
            num_test_cases = len(df)
            sum_test_cases = 0
            for i, row in df.iterrows():
                per_problem_rate = (row['n_unique_success'] / (row['n_unique_success'] + alpha)) * (100 + alpha)
                # print(per_problem_rate)
                sum_test_cases += per_problem_rate

            print(f"Problem set: {problem}\t Model: {model},\tMotifBenchScore: {sum_test_cases/num_test_cases:.2f}")
            if not os.path.exists(f'model-evals/scaffolding-results/scaffolding_summary.csv'):
                # write header to csv problem, model, score
                pd.DataFrame(columns=['problem_set', 'model_name', 'motif_bench_score']).to_csv(f'model-evals/scaffolding-results/scaffolding_summary.csv', index=False)
            # append result to csv 
            pd.DataFrame({'problem_set': [problem], 'model_name': [model], 'motif_bench_score': [sum_test_cases/num_test_cases]}).to_csv(f'model-evals/scaffolding-results/scaffolding_summary.csv', mode='a', header=False, index=False)


