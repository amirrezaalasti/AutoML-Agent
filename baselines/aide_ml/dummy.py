from baselines.aide_ml.aide_wrapper import AideWrapper
import os

# Set baseurl to https://chat-ai.cluster.uni-hannover.de/v1"
os.environ['OPENAI_BASE_URL'] = 'https://chat-ai.cluster.uni-hannover.de/v1'

def main():
    print("Starting experiment...")
    exp = AideWrapper(
        data_dir="example_tasks/bitcoin_price",  # replace this with your own directory
        goal="Build a time series forecasting model for bitcoin close price.",  # replace with your own goal description
        eval="RMSLE",  # replace with your own evaluation metric
        llm_model="llama-3.3-70b-instruct"
    )

    best_solution = exp.run(steps=2)

    print(f"Best solution has validation metric: {best_solution.valid_metric}")
    print(f"Best solution code: {best_solution.code}")
    print("Experiment finished.")

if __name__ == '__main__':
    main()