# GNN-for-OOKB  

#Requirements  
 chainer, cuda or numpy, more_itertools
 
#How to use
 1. run "python main.py"
 2. `head-1000` data will be used by default

#How to modify this model or develop your models  
 1. add your models in models dir.  
 2. register your models in models/manager.py      
 3. use option -nn to use your models, e.g., "python -nn X" runs the model X  

#How to analyse and investigate results   
 1. apply draw-score-history/draw.py to your results(scores) with thresholds including used margins in score functions.      
 2. this script shows an image (following image is an example), that is how the scores are changed in the learning. In particular, red and blue lines indicate negative and positive triplet's scores, respectively. the black line is your threshold, and the green line is accuracy using the threshold, i.e., how well the threshold splits triplets. this drawing is not the contribution of my paper, but i think it may help us to understand model's behavior.  

<img src="https://user-images.githubusercontent.com/17702908/33417466-e1fa11b4-d5e4-11e7-8bdd-6bf4f97325a8.png" width="600px">

#How to cite this work  
official paper: https://www.ijcai.org/proceedings/2017/0250.pdf  
official bibtex : https://www.ijcai.org/proceedings/2017/bibtex/250 (directly download the bibtex file)
