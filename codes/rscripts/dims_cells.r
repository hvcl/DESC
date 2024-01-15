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





folder <- paste0("../data2/", rds_name_only, "_", start_dim, "_", end_dim)


dimension_path = paste0(folder, "/dimensions.csv")


pbmc <- 0
tsnes <- 0

pbmc_path <- paste0(folder, "/", rds_name_only, "_pca.rds")


nrows <- 0

if(!file.exists(dimension_path))
{
	

	um <- Sys.umask(0)
	dir.create(folder, recursive=TRUE)
	Sys.umask(um)
	
	

	
	
	file_add <- paste0("rds/", rds_name)
	pbmc <- readRDS(file_add)
	
	pca_tf = "pca" %in% names(pbmc)
	
	if (pca_tf)
	{
		pca_dims <- length(pbmc@reductions$pca@stdev)
		
		if (pca_dims < end_dim){
			pbmc <- RunPCA(pbmc, verbose=FALSE, npcs=end_dim)
			saveRDS(pbmc, file=pbmc_path)
		} else{
			saveRDS(pbmc, file=pbmc_path)
		}		
	} else{
		pbmc <- RunPCA(pbmc, verbose=FALSE, npcs=end_dim)
		saveRDS(pbmc, file=pbmc_path)
	}
	
	

	

	pbmc_temp <- RunTSNE(object = pbmc, dims = 1:start_dim)
	
	pbmc_temp <- FindNeighbors(pbmc_temp, dims = 1:start_dim)
	pbmc_temp <- FindClusters(pbmc_temp, resolution = 0.5 ,verbose = FALSE)

	fp_df <- as.data.frame(Embeddings(object=pbmc_temp[["tsne"]])[, 1:2])


	x_name <- paste("x", start_dim, sep="")
	y_name <- paste("y", start_dim, sep="")
	zero <- c(0)
	  

	tsnes <- data.frame(row_col=zero, x_name=fp_df$tSNE_1, y_name=fp_df$tSNE_2)


	colnames(tsnes)[2] <- x_name
	colnames(tsnes)[3] <- y_name
	
	
	clusterings <- data.frame(new_dim = as.numeric(as.character(pbmc_temp@meta.data$seurat_clusters)))
	
	colnames(clusterings)[1] <- as.character(start_dim)
	
	

	count <- 4
	clu_count <- 2


	nrows <- nrow(fp_df)


	for (i in (start_dim+1):end_dim)
	{

		pbmc_temp <- RunTSNE(object = pbmc, dims = 1:i)
		
		pbmc_temp <- FindNeighbors(pbmc_temp, dims = 1:i)
		pbmc_temp <- FindClusters(pbmc_temp, resolution = 0.5 ,verbose = FALSE)

		fp_df <- as.data.frame(Embeddings(object=pbmc_temp[["tsne"]])[, 1:2])


		x_name <- paste("x", i, sep="")
		y_name <- paste("y", i, sep="")

		tsnes$x_name <- fp_df$tSNE_1
		tsnes$y_name <- fp_df$tSNE_2

		colnames(tsnes)[count] <- x_name
		colnames(tsnes)[count+1] <- y_name

		count <- count+2	


		clusterings$new_dim <- as.numeric(as.character(pbmc_temp@meta.data$seurat_clusters))
	
		colnames(clusterings)[clu_count] <- as.character(i)

		clu_count <- clu_count + 1
	}


	row_col <- paste(nrow(tsnes), "_", ncol(tsnes)-1, sep="")
	colnames(tsnes)[1] <- row_col

	

	write.csv(tsnes, file=dimension_path, row.names=FALSE)
	
	
	clu_path = paste0(folder, "/clusterings.csv")
	write.csv(clusterings, file=clu_path, row.names=FALSE)
	

	
	
} else {
	pbmc <- readRDS(pbmc_path)
	tsnes <- read.csv(file = dimension_path)
    
    nrows <- nrow(tsnes)
}



gene_names2 <- c()
final_genes <- ""
first <- TRUE

for (i in 1:length(gene_names))
{
	

	if (gene_names[i] %in% rownames(pbmc)) 
	{
		data.use <- (x = FetchData(
			object = pbmc,
			vars = gene_names[i],
			cells = NULL
		))


		data.gene <- na.omit(object = data.use)

		gene_vector <- data.gene[, gene_names[i]]

		k_test <- c()
		for (j in 1:length(gene_vector))
		{
			if (gene_vector[j] != 0)
			{
				k_test <- c(k_test, j)
			}

			if (length(k_test) == 3)
			{
				break
			}
		}
		
		
		
		if (length(k_test) == 3)
		{
			gene_names2 <- c(gene_names2, gene_names[i])
		
			if (first)
			{
				final_genes <- gene_names[i]
				
				first = FALSE
			} else {
				final_genes <- paste0(final_genes, "_", gene_names[i])
			}
		} 
	}
}


cat(final_genes)

gene_length = length(gene_names2)





if (gene_length > 0)
{
	data.use <- (x = FetchData(
	  object = pbmc,
	  vars = gene_names2,
	  cells = NULL
	))



	data.gene <- na.omit(object = data.use)




    
    
    cells_id = integer(gene_length*(1+nrows))
    
    

	for (i in 1:gene_length)
	{
		gene = gene_names2[i]
		
        gene_vector <- data.gene[, gene]
	
    
    
		ex_path <- paste0(folder, "/", gene, "_expression.csv")
		
		if (!file.exists(ex_path))
		{
		
            ex_df <- data.frame(ex = gene_vector)
			
			write.csv(ex_df, file=ex_path, row.names=FALSE)
		}
		
		
		

        id_count = 2
        
		for (j in 1:length(gene_vector))
        {
            if (gene_vector[j]!=0)
            {
                cells_id[(i-1)*(1+nrows) + id_count] = j-1
                
                id_count = id_count + 1
            }
        }
		
		
        cells_id[(i-1)*(1+nrows) + 1] = id_count - 2;
        
        
        
        
	
		
	}
    
    
    cat(",")
    cat(cells_id)
}











