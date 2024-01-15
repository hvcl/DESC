binpost  <- receiveBin()
rds_name <- rawToChar(binpost)


	
file_add <- paste0("rds/", rds_name)



if(file.exists(file_add)) {
	cat("true")
} else {
	cat("false")
}