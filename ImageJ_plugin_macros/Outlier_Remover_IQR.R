# input file pathway to be read
pathway = "/Applications/seeds_analyzer-master/project-intellij/Pictures/TestCrop/LabelledTestCrop.csv"

df1 = read.csv(pathway)

#  input number of species being recorded in csv
num_species = 3

# create new empty dataframe as required
dfNew = data.frame()

for (i in 1:num_species){
  species = readline()
  dfTemp = df1[df1$species==species, ]
  quartiles <- quantile(dfTemp$Area, probs=c(.25, .75), na.rm = FALSE)
  IQR <- IQR(data)
  
  Lower <- quartiles[1] - 1.5*IQR
  
  PixelOutlier <- 0.002
  
  Upper <- quartiles[2] + 1.5*IQR 
  
  dfTemp = dfTemp[dfTemp$Area > Lower & dfTemp$Area>PixelOutlier  & dfTemp$Area < Upper, ]
  dfNew = rbind(dfNew,dfTemp)
}
dfTest = df1[df1$species == 'C_acris' | df1$species == 'C_nigra' | df1$species == 'D_carota' , ]
dfNew$species

# Input path and file name of new csv
write.csv(dfNew, "/Applications/seeds_analyzer-master/project-intellij/Pictures/TestCrop/OutlierLabelledTestCrop.csv", row.names = FALSE )

