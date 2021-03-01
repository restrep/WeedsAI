# WeedsAI

A multiclass weeds image classifier based on the public dataset DeepWeeds. 

The model uses transfer learning and builds upon the ResNet50. Obtaining a test accuracy of 91%.

Check the [webapp](https://weedsai.herokuapp.com/) deployed using flask and heroku.

## Why?
The scenario where two plant species do not destroy each other is quite rare. Predominantly, plants will fight for all the available resources around them.

Weeds can be defined as plants that interfere with farming objectives or simply put, as plants growing where they are not wanted. In both developing and developed countries, weeds are the most limiting factor to agricultural production behind socioeconomic and crop management problems. In India, for example, weeds cost agricultural production more than 11 billion USD each year, and in Australia, it is estimated that farmers spend around 1.5 billion AUS each year on weed control.

The control of weeds is a time-consuming and intense task typically performed by humans. Improving the success and speed of weed control through robotics will have a tremendous increase in agricultural productivity. One key factor in the use of robotics for weed control is identifying weeds accurately and early as possible.

WeedsAI is multiclass weeds image classifier based on the public dataset DeepWeeds. The dataset consists of 17,509 images of plants belonging to 8 different weeds species native to different locations across Australia

## How?
WeedsAI uses transfer learning leveraging the power of the successful pre-trained model ResNet50 and building a deep neural network on top of it obtaining 91% accuracy.

Here is the confusion matrix (%) for a 20% test set.

For more details check the ![jupyter notebook](https://github.com/restrep/WeedsAI/blob/main/WeedsAI_Notebook.ipynb).

![](https://github.com/restrep/WeedsAI/blob/main/Confussion%20Matrix.png)
                    
     
