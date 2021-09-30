binpost  <- receiveBin()
postdata <- rawToChar(binpost)


	

postdata2 <- strsplit(postdata, "_")



start_dim <- as.numeric(as.character(postdata2[[1]][1]))
end_dim <- as.numeric(as.character(postdata2[[1]][2]))
rds_name <- postdata2[[1]][3]
rds_name_only <- gsub(".rds", "", rds_name)


gene_names <- c()
for (i in 4:length(postdata2[[1]]))
{
	gene_names <- c(gene_names, postdata2[[1]][i])
}





# folder <- paste0("../data2/", start_dim, "_", end_dim)
folder <- paste0("../data2/", rds_name_only, "_", start_dim, "_", end_dim)

# pbmc_path <- paste0(folder, "/pbmc_pca.rds")
pbmc_path <- paste0(folder, "/", rds_name_only, "_pca.rds")
pbmc <- readRDS(pbmc_path)

	

	



data.use <- (x = FetchData(
  object = pbmc,
  vars = gene_names,
  cells = NULL
))



data.gene <- na.omit(object = data.use)



for (i in 1:length(gene_names))
{
	gene = gene_names[i]
	
	
	
	
	ex_path <- paste0(folder, "/", gene, "_expression.csv")
	
	if (!file.exists(ex_path))
	{
		ex_df <- data.frame(ex = data.gene[, gene])
		
		write.csv(ex_df, file=ex_path, row.names=FALSE)
	}
}








