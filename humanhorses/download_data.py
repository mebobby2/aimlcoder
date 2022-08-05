import urllib.request
import zipfile

# Downloading trainig data

# url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip"

# file_name = "horse-or-human.zip"
# training_dir = 'horse-or-human/training/'
# urllib.request.urlretrieve(url, file_name)

# zip_ref = zipfile.ZipFile(file_name, 'r')
# zip_ref.extractall(training_dir)
# zip_ref.close()

# Downloading validation data
validation_url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip"

validation_file_name = "validation-horse-or-human.zip"
validation_dir = 'horse-or-human/validation/'
urllib.request.urlretrieve(validation_url, validation_file_name)

zip_ref = zipfile.ZipFile(validation_file_name, 'r')
zip_ref.extractall(validation_dir)
zip_ref.close()
