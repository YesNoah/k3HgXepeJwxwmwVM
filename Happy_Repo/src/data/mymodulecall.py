import importlib
import datautils
from datautils import import_data

importlib.reload(import_data)
df=import_data.CSVtoDF()
df.head()
from datautils import processing
importlib.reload(processing)
X_train, X_test, y_train, y_test= processing.prepro(df)
