def replace_spaces_with_underscore(input_file, output_file):
    try:
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            for line in infile:
                # Replace spaces with "__" and write to the output file
                modified_line = line.replace(' ', '__')
                outfile.write(modified_line)
    except FileNotFoundError:
        print("Input file not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")



# Example usage:
dir = 'en_be/dev.tok.norm.en'
replace_spaces_with_underscore(dir, dir)

dir = 'en_be/dev.tok.norm.es'
replace_spaces_with_underscore(dir, dir)

# d