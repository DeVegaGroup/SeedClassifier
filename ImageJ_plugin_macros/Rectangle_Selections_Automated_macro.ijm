input_directory = "/Users/ashworth/OneDrive - Norwich Bioscience Institutes/Jonathan_PhD_project/Rebonto/Seed_Images + CSV/Jonathan_test_raw/comparing_providers/"
output_directory = "/Users/ashworth/OneDrive - Norwich Bioscience Institutes/Jonathan_PhD_project/Rebonto/Seed_Images + CSV/Jonathan_test_cropped/comparing_providers/"
function action(input, output, filename) {
        open(input + filename);
        makeRectangle(360, 280, 4472, 6600);
        run("Crop");
        saveAs("Tiff", output + filename);
        close();
}

setBatchMode(true); 
list = getFileList(input_directory);
for (i = 0; i < list.length; i++){
        action(input_directory, output_directory, list[i]);
}
setBatchMode(false);

// original coordinates:- 584, 392, 4128, 6504