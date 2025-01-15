# Define the path to the text file containing arguments
args_file_path = "ru_index_mapping.txt"

# Define the command you want to run with changing arguments
command_template = "python CSI_doppler_computation_final_grouping.py ./processed_phase/ S1a,S2a,S4a,S5a ./doppler_traces/ 400 400 10 1 -1.2 RU{} {} {} 10"

# Open the text file and read the lines
with open(args_file_path, "r") as args_file:
    lines = args_file.readlines()

# Iterate through the lines and run the command with each pair of arguments
for i, line in enumerate(lines):
    # Extract two arguments from the current line
    # print(i, line)
    # arg1, arg2 = map(float, line.strip().split())
    line = line.strip().split()
    line[0] = int(line[0].strip(","))
    line[1] = int(line[1].strip(","))
    arg1, arg2 = line[0], line[-1]

    # Construct the command with the arguments
    command = command_template.format(int(i+1), arg1, arg2)

    # Print the command for reference
    # print(f"Running command {i + 1}: {command}")

    # Uncomment the following line to actually run the command
    import subprocess
    subprocess.call(command, shell=True)
