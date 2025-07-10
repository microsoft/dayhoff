import pandas as pd 
import os 
import glob
import shutil 

models = ['dayhoff-3b', 'dayhoff-170m', 'evodiff', 'jamba-170m-10mbothfilter-36w/', 'jamba-170m-10mnofilter-36w/', 'jamba-170m-gigaclust-36w/', 'jamba-170m-seq-36w/', 'jamba-170m-seqsam-36w/', 'jamba-3b-indel-gigaclust-120k-2/', 'jamba-3b-seq-sam-biar-fsdp-tok90k/']
problem_set = ['motifbench', 'rfdiff']

model_dict = {'dayhoff-3b': '3b-cooled', 
              'dayhoff-170m': '170m-ur50-bbr-sc', #jamba-170m-10mrmsd-36w
              'evodiff': 'evodiff', 
              'jamba-170m-10mbothfilter-36w/': '170m-ur50-bbr-n', 
              'jamba-170m-10mnofilter-36w/': '170m-ur50-bbr-u', 
              'jamba-170m-gigaclust-36w/': '170m-ggr',
              'jamba-170m-seq-36w/': '170m-ur50', 
              'jamba-170m-seqsam-36w/': '170m-ur90', 
              'jamba-3b-indel-gigaclust-120k-2/': '3b-ggr-msa', 
              'jamba-3b-seq-sam-biar-fsdp-tok90k/': '3b-ur90'}

zenodo_dir = f'model-evals/scaffolding-results/scaffolding_zenodo/'

mega_df = []
alpha = 5

for problem in problem_set:
    for model in models: 
        if not os.path.exists(os.path.join(zenodo_dir, problem, 'results', model_dict[model])):
            os.makedirs(os.path.join(zenodo_dir, problem, 'results', model_dict[model]), exist_ok=True)
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
            mega_df.append({'problem': problem, 'model': model, 'MotifBenchScore': sum_test_cases/num_test_cases})
        
        individual_result_files = sorted(glob.glob(f'model-evals/scaffolding-results/{problem}/{model}/*_unique.csv'))
        for individual_file in individual_result_files:
            #copy file into zenodo directory
            shutil.copy(individual_file, os.path.join(zenodo_dir, problem, 'results', model_dict[model], os.path.basename(individual_file)))

# save results to csv 
if not os.path.exists(f'model-evals/scaffolding-results/scaffolding_summary.csv'):
    pd.DataFrame(columns=['problem', 'model', 'MotifBenchScore']).to_csv(f'model-evals/scaffolding-results/scaffolding_summary.csv', index=False)




