import subprocess

def run_container(image_name, input_path, output_path):
    command = [
        "docker", "run", 
        "-e", f"INPUT_PATH={input_path}", 
        "-e", f"OUTPUT_PATH={output_path}", 
        image_name
    ]
    subprocess.run(command, check=True)

def run_container2(image_name, input_path, output_path, model_path):
    command = [
        "docker", "run", 
        "-e", f"INPUT_PATH={input_path}", 
        "-e", f"OUTPUT_PATH={output_path}", 
        "-e", f"MODEL_PATH={model_path}", 
        image_name
    ]
    subprocess.run(command, check=True)

def run_container3(image_name, model_path):
    command = [
        "docker", "run", 
        "-e", f"MODEL_PATH={model_path}", 
        image_name
    ]
    subprocess.run(command, check=True)

def main():
    # Define paths for input and output data
    input_path1 = r"C:\Users\proba\OneDrive\Documents\Booz Training\generative-ai-audio-enhancement-classification\data\original"
    output_path1 = r"C:\Users\proba\OneDrive\Documents\Booz Training\generative-ai-audio-enhancement-classification\data\augmented"

    output_path2 = r"C:\Users\proba\OneDrive\Documents\Booz Training\generative-ai-audio-enhancement-classification\data\processed"

    output_path3 = r"C:\Users\proba\OneDrive\Documents\Booz Training\generative-ai-audio-enhancement-classification\data\features"

    model_path_test = r"C:\Users\proba\OneDrive\Documents\Booz Training\generative-ai-audio-enhancement-classification\models\docker_model.keras"
    model_path_save = r"C:\Users\proba\OneDrive\Documents\Booz Training\generative-ai-audio-enhancement-classification\models\docker_model_trained.keras"
    # Run container 1
    run_container("data_aug_docker", input_path1, output_path1)

    # Run container 2 using output from container 1
    run_container("preproc_docker", output_path1, output_path2)

    # Run container 3 using output from container 2
    run_container("feat_extract_docker", output_path2, output_path3)

    # Run container 4 using output from container 3
    run_container3("model_docker", model_path_test)

    # Run container 5 using output from container 4
    run_container("training_docker", output_path3, model_path_save, model_path_test)

if __name__ == "__main__":
    main()
