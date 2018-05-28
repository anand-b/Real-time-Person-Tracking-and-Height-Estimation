# Real-time Person tracking and Height Estimation
**Developer:** Anand Balan

The objective of this project is to detect a human being wearing a T-shirt with particular logo and estimate the height of the person in feet and inches. The project folder contains two files - main.py and track_any.py. main.py file can be used to detect and track a single person wearing the specific t-shirt. track_any.py can be used to detect and track any person. However, for both the programs to work, there must be atleast one person wearing the specific t-shirt.

This is because, the project uses the logo as the reference object to estimate the height of the person.

##### HOW TO RUN THE PROGRAM?
1. Open a terminal / command prompt
2. Run the following command:
	For single person detection - python main.py 
	For multiple person detection - python track_any.py
	(or)
	For single person detection - python3 main.py 
	For multiple person detection - python3 track_any.py


On the output live stream, 
- The person detected and tracked will be enclosed in a box. 
- The logo detected will be enclosed in a box.
- The eyes and face of the person will be enclosed in a circle and box respectively.
- The estimated height of the person will be shown in the video itself.