pathway = "/Applications/seeds_analyzer-master/project-intellij/Pictures/TestCrop/TestCrop.csv"
df1 = read.csv(pathway)
df1$species <- substr(as.character(df1$species),
                                      start= 1, 
                                      stop= nchar(as.character(df1$species) )-1 )
write.csv(df1, "/Applications/seeds_analyzer-master/project-intellij/Pictures/TestCrop/LabelledTestCrop.csv", row.names = FALSE)
