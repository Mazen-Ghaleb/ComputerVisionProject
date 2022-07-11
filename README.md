# Simple Perception Stack for Self-Driving Cars

### How to enable debugger:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Change the value of debugger to true

### Methods to choose from:

1. Line fitting Method
2. Curve fitting Method
3. Point fitting Method
4. All Methods

### How to run:

1. Run the Shell
2. Run the Python File in VScode or any IDE
3. Run through the Jupyter notebook

# Description

The pipeline was divided into two major phases : Lane Detectetion and Object Detection (Example: Cars)

## a) Object Detection

This phase was completed by using yolov3 with coco names to detect objects within the frame/image.

## b) Lane Detection

This phase was completed on the following step on the original frame/image then combining it with the object detection to get the result. <br/>

Original Image
![](./media/read_me/test4.jpg)

1. Apply Gaussian Blur to Image <br />
   ![](./media/read_me/Gaussian.png)

2. Changing the Image to HLS color <br />
   ![](./media/read_me/HSL.png)

3. Using S channel of the Image <br />
   ![](./media/read_me/S%20channel.png)

4. Applying Canny Edge Detection on the Image <br />
   ![](./media/read_me/Canny.png)

5. Applying Sobel to remove Horizontal Edges <br />
   ![](./media/read_me/Sobel.png)

6. Scaling Sobel to Color Scale <br />
   ![](./media/read_me/Scaled%20Sobel.png)

7. Thresholding the Image to clear some noise <br />
   ![](./media/read_me//Thresholded%20Sobel.png)

8. Apply Openning if needed (too much noise) <br />
   Note: In this step, we used Skimage Opening, to provide smoother image with less noise however it heavily reduces the performance. <br/>

   Comparison between Opening in both:
   |Open CV | Skimage|
   |:-:|:-:|
   |![](./media/read_me/Opening.png) | ![](./media/read_me/OpeniningSKI.png) |

9. Apply Algorithim to get Line Points

   |           Left Lane Points           |           Right Lane Points           |
   | :----------------------------------: | :-----------------------------------: |
   | ![](./media/read_me/Left%20Lane.png) | ![](./media/read_me/Right%20Lane.png) |

10. Combine it according to the method chosen
    | Line Fitting Method | Curve Fitting Method | Point Fitting Method |
    |:-:|:-:|:-:|
    | ![](./media/read_me/Samples/Image%204%20-%20Line%20Method.png) | ![](./media/read_me/Samples/Image%204%20-%20Curve%20Method.png) | ![](./media/read_me/Samples/Image%204%20-%20Point%20Method.png) |

# Test Images with all methods

|               Original Image               |                      Line Fitting Method                       |                      Curve Fitting Method                       |                      Point Fitting Method                       |
| :----------------------------------------: | :------------------------------------------------------------: | :-------------------------------------------------------------: | :-------------------------------------------------------------: |
| ![](./media/read_me/Samples/Image%201.png) | ![](./media/read_me/Samples/Image%201%20-%20Line%20Method.png) | ![](./media/read_me/Samples/Image%201%20-%20Curve%20Method.png) | ![](./media/read_me/Samples/Image%201%20-%20Point%20Method.png) |
| ![](./media/read_me/Samples/Image%202.png) | ![](./media/read_me/Samples/Image%202%20-%20Line%20Method.png) | ![](./media/read_me/Samples/Image%202%20-%20Curve%20Method.png) | ![](./media/read_me/Samples/Image%202%20-%20Point%20Method.png) |
| ![](./media/read_me/Samples/Image%203.png) | ![](./media/read_me/Samples/Image%203%20-%20Line%20Method.png) | ![](./media/read_me/Samples/Image%203%20-%20Curve%20Method.png) | ![](./media/read_me/Samples/Image%203%20-%20Point%20Method.png) |
| ![](./media/read_me/Samples/Image%204.png) | ![](./media/read_me/Samples/Image%204%20-%20Line%20Method.png) | ![](./media/read_me/Samples/Image%204%20-%20Curve%20Method.png) | ![](./media/read_me/Samples/Image%204%20-%20Point%20Method.png) |
| ![](./media/read_me/Samples/Image%205.png) | ![](./media/read_me/Samples/Image%205%20-%20Line%20Method.png) | ![](./media/read_me/Samples/Image%205%20-%20Curve%20Method.png) | ![](./media/read_me/Samples/Image%205%20-%20Point%20Method.png) |
| ![](./media/read_me/Samples/Image%206.png) | ![](./media/read_me/Samples/Image%206%20-%20Line%20Method.png) | ![](./media/read_me/Samples/Image%206%20-%20Curve%20Method.png) | ![](./media/read_me/Samples/Image%206%20-%20Point%20Method.png) |
| ![](./media/read_me/Samples/Image%207.png) | ![](./media/read_me/Samples/Image%207%20-%20Line%20Method.png) | ![](./media/read_me/Samples/Image%207%20-%20Curve%20Method.png) | ![](./media/read_me/Samples/Image%207%20-%20Point%20Method.png) |
| ![](./media/read_me/Samples/Image%208.png) | ![](./media/read_me/Samples/Image%208%20-%20Line%20Method.png) | ![](./media/read_me/Samples/Image%208%20-%20Curve%20Method.png) | ![](./media/read_me/Samples/Image%208%20-%20Point%20Method.png) |
