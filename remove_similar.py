import os
import shutil
n= 'C:/Users/Hanzalah Choudhury/Desktop/boundingbox/similar.txt'

source_folder = 'C:/Users/Hanzalah Choudhury/Desktop/boundingbox/data/bb/'
destination_folder = 'C:/Users/Hanzalah Choudhury/Desktop/boundingbox/data/bb/similar_discard/'

with open(n, 'r') as f:
    data= f.readlines()

print(data[len(data)-1].split(',')[0])
for i in range(1, len(data)-1):
        nmae = data[i].split(',')[0].strip()
        source_path = os.path.join(source_folder, nmae)
        try:
            destination_path = os.path.join(destination_folder, nmae)
            shutil.move(source_path, destination_path)
        except:
             pass