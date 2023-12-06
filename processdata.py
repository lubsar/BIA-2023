import pandas as pd

file_names = ['de.csv', 'pso.csv', 'soma.csv', 'firefly.csv', 'tlbo.csv']

algorithm_data = {}

for file_name in file_names:
    algo_df = pd.read_csv(file_name)
    algo_df.set_index('experiment', inplace=True)
    
    algorithm_name = file_name.split('.')[0] 
    algorithm_data[algorithm_name] = algo_df

excel_writer = pd.ExcelWriter('results.xlsx', engine='xlsxwriter')

for test_function in algorithm_data['de'].columns:
    print(f"Test Function: {test_function}\n")
    
    table_data = {algo_name: algo_df[test_function] for algo_name, algo_df in algorithm_data.items()}
    table_df = pd.DataFrame(table_data)
    table_df.to_excel(excel_writer, sheet_name=test_function)
    
    print(table_df)
    print("\n")

excel_writer.save()

merged_data = pd.concat(algorithm_data.values(), axis=1, keys=algorithm_data.keys())
excel_writer = pd.ExcelWriter('merged.xlsx', engine='xlsxwriter')
merged_data.to_excel(excel_writer, sheet_name='All_Test_Functions', index_label='experiment')
excel_writer.save()