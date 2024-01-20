# Importing all necessary libraries
from tkinter import *
import tkinter as tk

import cv2, os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk


# Creating the main Tkinter window
root = Tk()

# Setting the width and height of the window
window_width = 600
window_height = 600

# Getting the screen width and height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Calculating the x and y coordinates for the window to be centered
x_coordinate = (screen_width - window_width) // 2
y_coordinate = (screen_height - window_height) // 2

# Set the geometry of the window
root.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

# Set the title of the window
root.title('Facial Attendance System')

# Create a label in the root window
lbl1 = Label(root, fg="blue", text="Facial Attendance System", font=("Arial Bold", 20), pady=10)
lbl1.grid(row=1, column=1, pady=(20, 50), padx=140)

# Define global variables Id and name
global Id, name

# path to xml file
harcascadePath = "haarcascade_frontalface_default.xml"


# Function to open the registration window
def reg_window():
    root.iconify()
    reg = Tk()
    reg.title('Register')

    # Set the width and height of the registration window
    window_width = 600
    window_height = 600

    # Get the screen width and height
    screen_width = reg.winfo_screenwidth()
    screen_height = reg.winfo_screenheight()

    # Calculate the x and y coordinates for the window to be centered
    x_coordinate = (screen_width - window_width) // 2
    y_coordinate = (screen_height - window_height) // 2

    # Set the geometry of the registration window
    reg.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

    # creating labels for Name entry and ID entry
    nameLabel = Label(reg, text="Enter Name", pady=40, padx=20, font=("Arial", 14))
    idLabel = Label(reg, text="Enter ID", pady=5, padx=20, font=("Arial", 14))

    # placing the labels in the screen
    nameLabel.grid(column=0, row=0)
    idLabel.grid(column=0, row=1)

    # creating textboxes for entry of name and id
    nameEntry = Entry(reg, font=("Arial", 14))
    nameEntry.grid(column=1, row=0)

    idEntry = Entry(reg, font=("Arial", 14))
    idEntry.grid(column=1, row=1)

    # Function to get details from the user
    def get_details():
        Id = (idEntry.get())
        name = (nameEntry.get())
        return Id, name

    # Function to capture images for training
    def TakeImages():

        # getting student details
        Id, name = get_details()

        # creating a videoCapture object to capture video from webcam
        cam = cv2.VideoCapture(0)

        # allows to use to detect face in images or video frames
        detector = cv2.CascadeClassifier(harcascadePath)

        # count of images
        sampleNum = 0

        
        while (True):

            # reading a frame from video capture object 'cam' and 
            # assigning to variables ret and img
            ret, img = cam.read()

            # converting the captured color image to grayscale to simplify processing
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # uses detector object to detect faces in grayscale image
            faces = detector.detectMultiScale(gray, 1.3, 5)

            # marking the detected faces with blue rectangles on the original image
            # and saving the face regions as individual image files
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                sampleNum = sampleNum + 1
                cv2.imwrite("TrainingImage\ " + name + "." + Id + '.' + str(sampleNum) + ".jpg",
                            gray[y:y + h, x:x + w])
                cv2.imshow('Taking Images', img)

            # waiting user to intervene or if sample >=50, automatically breaks
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif sampleNum >= 50:
                break
        
        # release the video capture object
        cam.release()

        # closing all OpenCV windows        
        cv2.destroyAllWindows()

        # creating a list 'row' containing student ID and name
        row = [Id, name]

        # opeining the csv file in reading and appending mode
        with open("StudentDetails\StudentDetails.csv", 'a+') as csvFile:
            # creating a CSV writer object
            writer = csv.writer(csvFile)
            # writing the row list to the csv file
            writer.writerow(row)
            # closing the csv file
            csvFile.close()

        # printing images taken to the console
        print("Images Taken")
        # creating a label to show images taken
        success = Label(reg, text="Images Taken", font=("Arial Bold", 15), padx=70)
        # placing the success label to the grid
        success.grid(column=1, row=7, columnspan=2)

    # Function to clear the registration window
    def clear_reg():
        # restore root window to its previous state
        root.deiconify()
        # destroying the reg window
        reg.destroy()

    # Function to train the captured images
    def TrainImages():
        # Create an LBPH (Local Binary Pattern Histogram) face recognizer
        recognizer = cv2.face_LBPHFaceRecognizer.create()

        # Call a function (possibly defined elsewhere) to get images and corresponding labels for training
        faces, Id = get_images_and_data("TrainingImage")

        # Train the recognizer with the provided faces and labels
        recognizer.train(faces, np.array(Id))

        # Save the trained recognizer model to a file
        recognizer.save("TrainingImageyml\Trainner.yml")

        # Create a Label widget indicating that the images have been trained
        success = Label(reg, text="Images Trained Successfully", font=("Arial Bold", 15), padx=70)

        # Place the 'success' Label widget in the grid layout of the 'reg' window
        success.grid(column=1, row=8, columnspan=2)


    # Function to retrieve images and data for training
    def get_images_and_data(path):
        # Create a list of file paths by joining the 'path' directory with each file in that directory
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

        # Initialize empty lists to store face images and corresponding labels
        faces = []
        Ids = []

        # Iterate through each file path in the list
        for paths in imagePaths:
            # Open the image using the Python Imaging Library (PIL) and convert it to grayscale ('L')
            pil_image = Image.open(paths).convert('L')

            # Convert the PIL image to a NumPy array of unsigned 8-bit integers ('uint8')
            imageNp = np.array(pil_image, 'uint8')

            # Extract the label (Id) from the file name using os.path functions
            Id = int(os.path.split(paths)[-1].split(".")[1])

            # Append the NumPy array of the face image and its corresponding label to the lists
            faces.append(imageNp)
            Ids.append(Id)

        # Return the lists of face images and labels
        return faces, Ids

    # Button to trigger the 'TakeImages' function
    btnTakeImg = Button(reg, text='Take Image', command=TakeImages, height=2, width=20, font=("Arial", 14))
    btnTakeImg.grid(row=3, column=1, columnspan=3, padx=35, pady=30)

    # Button to trigger the 'TrainImages' function
    btnTrainImg = Button(reg, text='Train Image', command=TrainImages, height=2, width=20, anchor=CENTER, font=("Arial", 14))
    btnTrainImg.grid(row=4, column=1, columnspan=2, padx=35, pady=0)

    # Button to trigger the 'clear_reg' function, with "GO BACK" text, red foreground, light grey background
    btnClear = Button(reg, text='GO BACK', fg="red", bg="lightgrey", command=clear_reg, height=2, width=15, anchor=CENTER, font=("Arial", 12))
    btnClear.grid(row=5, column=1, columnspan=2, padx=35, pady=40)

    # Start the tkinter main event loop
    reg.mainloop()

# Function to mark attendance
def attend():

    # creating LBPH face recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # load the trained recognizer model
    recognizer.read("TrainingImageyml\Trainner.yml")
    #load the Haar Cascade classifier for face detection
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    # reading the student details from the csv
    df = pd.read_csv("StudentDetails\StudentDetails.csv")
    # opening the video camera
    cam = cv2.VideoCapture(0)
    # set the font for drawing on the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    # define column names for attendance DataFrame
    col_names = ['Id', 'Name', 'Date', 'Time']
    # create DataFrames for attendance and attendance details
    attendance = pd.DataFrame(columns=col_names)
    at_details = pd.DataFrame(columns=col_names)
    
    # infinite loop for capturing video frames
    while True:
        # capture a frame from the camera
        ret, im = cam.read()
        # converting frame to grayscale
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # detecting face in grayscale frame
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)

        #process each detected face
        for (x, y, w, h) in faces:
            # drawing a rectange around the face
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            # predict the ID of the face
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            
            # check confidence level and mark attendance
            if (conf < 50):
                # getting current timestamp
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                
                # getting name corresponding to predicted id
                aa = df.loc[df['Id'] == Id]['Name'].values

                # creating a string for displaying and updating id from the frame
                tt = str(Id) + "-" + aa
                attendance.loc[len(attendance)] = [Id, aa, date, timeStamp]
                at_details.loc[len(at_details)] = [Id, aa, date, timeStamp]
            else:
                Id = 'Unknown'
                tt = str(Id)

            # draw the ID on the image
            cv2.putText(im, str(tt), (x, y + h), font, 1, (255, 255, 255), 2)

        # removing duplicate entrites in the attendance dataframes
        at_details = at_details.drop_duplicates(subset=['Id'], keep='first')
        attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
        
        cv2.imshow('Taking Images', im)

        # breaking the loop if 'q' key is pressed
        if (cv2.waitKey(1) == ord('q')):
            break
    

    # getting timestamp for file naming
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour, Minute, Second = timeStamp.split(":")
    
    # creating file names for attendance CSV files
    fileName = "Attendance\Attendance_" + date + "_" + Hour + ".csv"

    # check if file exists
    if(os.path.exists(fileName)):
        attendance.to_csv(fileName, mode='a', index=False, header=False)
    else:
        # save attendance DataFrames to CSV files
        attendance.to_csv(fileName, index=False)

    fileName2 = "Attendance\Attendance.csv"
    # check if file exists
    if (os.path.exists(fileName2)):
        at_details.to_csv(fileName2, mode='a', index=False, header=False)
    else:
        # save attendance DataFrames to CSV files
        at_details.to_csv(fileName2, index=False)

    cam.release()
    cv2.destroyAllWindows()
    
    # creating a success label in the root window
    success = Label(root, text="Attendance Successful", font=("Arial Bold", 14), padx=50)
    success.grid(column=1, row=7, pady=20, columnspan=2)
    print("success")

# Function to display attendance details
def attend_details():
    details_window = Tk()
    details_window.title('Attendance details')

    # specifying window height and width
    width = 500
    height = 400
    screen_width = details_window.winfo_screenwidth()
    screen_height = details_window.winfo_screenheight()
    x = (screen_width / 2) - (width / 2)
    y = (screen_height / 2) - (height / 2)
    details_window.geometry("%dx%d+%d+%d" % (width, height, x, y))
    details_window.resizable(0, 0)

    # putting table for attendance details
    TableMargin = Frame(details_window, width=500)
    TableMargin.pack(side=TOP)
    scrollbarx = Scrollbar(TableMargin, orient=HORIZONTAL)
    scrollbary = Scrollbar(TableMargin, orient=VERTICAL)
    tree = ttk.Treeview(TableMargin, columns=("Id", "Name", "Date", "Time"), height=400, selectmode="extended",
                        yscrollcommand=scrollbary.set, xscrollcommand=scrollbarx.set)
    scrollbary.config(command=tree.yview)
    scrollbary.pack(side=RIGHT, fill=Y)
    scrollbarx.config(command=tree.xview)
    scrollbarx.pack(side=BOTTOM, fill=X)
    tree.heading('Id', text="Id", anchor=W)
    tree.heading('Name', text="Name", anchor=W)
    tree.heading('Date', text="Date", anchor=W)
    tree.heading('Time', text="Time", anchor=W)
    tree.column('#0', stretch=NO, minwidth=0, width=0)
    tree.column('#1', stretch=NO, minwidth=0, width=100)
    tree.column('#2', stretch=NO, minwidth=0, width=100)
    tree.column('#3', stretch=NO, minwidth=0, width=100)
    tree.pack()

    # reading and giving the total attendance counts
    with open('Attendance/Attendance.csv') as f:
        reader = csv.DictReader(f, delimiter=',')
        row_count = 0
        for row in reader:
            Id = row['Id']
            Name = row['Name']
            Date = row['Date']
            Time = row['Time']
            tree.insert("", 0, values=(Id, Name[2:-2], Date, Time))
            row_count += 1
            # count = len(row["Id"])

    # inserting total attendance row
    tree.insert("", tk.END, values=("Total attendance", row_count))

    details_window.mainloop()

# Create buttons for attendance, registration, and attendance details
font = ("Arial", 14)
btn1 = Button(root, text='Attend', height=2, width=20, command=attend, anchor=CENTER, font=font)
btn1.grid(row=3, column=1, padx=80, pady=10)
btn2 = Button(root, text='Register', height=2, width=20, command=reg_window, font=font)
btn2.grid(row=4, column=1, padx=5, pady=10)
btn3 = Button(root, text="Attendance Details", height=2, width=20, command=attend_details, font=font)
btn3.grid(row=5, column=1, padx=5, pady=10)

# Start the main loop
root.mainloop()
