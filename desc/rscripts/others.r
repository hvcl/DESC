binpost  <- receiveBin()
postdata <- rawToChar(binpost)

postdata2 <- strsplit(postdata, ",")


start_dim <- as.numeric(as.character(postdata2[[1]][1]))
end_dim <- as.numeric(as.character(postdata2[[1]][2]))
rds_name <- postdata2[[1]][3]
rds_name_only <- gsub(".rds", "", rds_name)
selected_dim <- as.numeric(as.character(postdata2[[1]][4])) + start_dim

selected_gene <-  strsplit(postdata2[[1]][5], "_") 

# selected_label <- as.numeric(as.character(postdata2[[1]][6]))
cell_ids_split <- strsplit(postdata2[[1]][6], "_")

cell_ids <- c()
for (i in 1:length(cell_ids_split[[1]]))
{
	cell_ids <- c(cell_ids, (as.numeric(as.character(cell_ids_split[[1]][i]))+1))
}



# folder <- paste0("../data2/", start_dim, "_", end_dim)
folder <- paste0("../data2/", rds_name_only, "_", start_dim, "_", end_dim)

# pbmc_path <- paste0(folder, "/pbmc_pca.rds")
pbmc_path <- paste0(folder, "/", rds_name_only, "_pca.rds")
pbmc <- readRDS(pbmc_path)

# dbscan_path <- paste0(folder, "/", selected_gene, "_dbscan.csv")
# dbscan <- read.csv(file = dbscan_path)

# cells_path <- paste0(folder, "/", selected_gene, "_cells.csv")
# cells_df <- read.csv(file = cells_path)
# cells <- cells_df[["cells"]]



# cur_dim <- paste0("l", selected_dim)
# dim_labels <- dbscan[[cur_dim]]
# label_ids <- which(dim_labels==selected_label)
# label_cells <- cells[label_ids]






gene_names <- rownames(pbmc)

data.use <- (x = FetchData(
  object = pbmc,
  vars = gene_names,
  cells = NULL
))

data.gene <- na.omit(object = data.use)

# cells_data <- data.gene[label_cells, ]
cells_data <- data.gene[cell_ids, ]

# nrows <- nrow(data.gene)
cells_len <- length(cell_ids)

ratio <- 0.7


# candidates <- c()
final_genes <- c()
for (i in 1:ncol(cells_data))
{
	gene <- cells_data[, i]
	gene_name <- colnames(cells_data)[i]
    
    gene_count <- 1
    
    for (j in 1:length(selected_gene[[1]]))
    {
        if (gene_name==selected_gene[[1]][j])
        {
            gene_count <- 0
            
            break
        }
    }
    
	if (gene_count==1)
    {
        cur_cell_ids = which(gene>0)
        
        if (length(intersect(cell_ids, cur_cell_ids)) > (length(cur_cell_ids)*ratio))
            final_genes <- c(final_genes, gene_name)
    }
    
    
	# if ((length(which(gene>0)) > (cells_len*ratio)) & (gene_name!=selected_gene))
    # if ((length(which(gene>0)) > (cells_len*ratio)) & (gene_count==1))
    # {
		# candidates <- c(candidates, gene_name)
        
        
	# }
}


# final_genes <- c()
# for (i in 1:length(candidates))
# {
	# gene_vector <- data.gene[, candidates[i]]
	
	# k_results <- kmeans(gene_vector, 5)

	# k_df <- data.frame(gene=gene_vector, k=k_results[[1]])
	
	
	
	# clusters <- list()
	# mins <- c()
	
	# for (j in 1:5)
	# {
		# clusters[[j]] <- subset(k_df, k==j)
		# mins[j] <- min(clusters[[j]]$gene)
	# }
	
	# s <- sort(mins)
	# indices <- c()
	
	# for (j in 1:5)
	# {
		# if (mins[j] >= s[4])
		# {
			# indices <- c(indices, j)
		# }
	# }
	
	
	# ex_cells <- rbind(clusters[[indices[1]]], clusters[[indices[2]]])
	# cells_id <- as.numeric(as.character(rownames(ex_cells)))
	

	# inter_num <- length(intersect(cell_ids, cells_id))

	# if (inter_num > (length(cells_id)*ratio))
		# final_genes <- c(final_genes, candidates[i])
# }

cat(final_genes)







