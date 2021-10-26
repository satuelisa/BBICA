# delete JSON files over two days old
for file in `find *.json -maxdepth 1 -mtime +2 -type f`;
do
    rm $file
done
	    
