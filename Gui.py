import tkinter as tk
from tkinter import scrolledtext
import subprocess

# Define function to handle button click
def prediction_button_click():
    # Get value from input box 1
    division = input_box1.get()

    # Call the Python script with the division as an argument
    subprocess.call(['python', 'Bayesian\Predict_ver1.py', division])

def data_scrape():
    # URL=args.string1
    # output_path = args.string2
    # mode= args.string3

    # Get value from input box 2
    URL = input_box2.get()
    # Get value from input box 3
    output_path = input_box3.get()
    # Get value from input box 4
    mode = input_box4.get()
    # Call the Python script with the division as an argument
    subprocess.call(['python', 'Data_scraper\SoccerWay.py', URL,output_path,mode,"1"])
    
    
# Define function to display tooltip
def show_tooltip(widget, text):
    x, y, cx, cy = widget.bbox("insert")
    x = widget.winfo_rootx() + x + cx / 2
    y = widget.winfo_rooty() + y + cy + 10
    # Create a tooltip window
    tw = tk.Toplevel(widget)
    # Remove window decorations
    tw.wm_overrideredirect(True)
    # Set window position
    tw.wm_geometry("+%d+%d" % (x, y))
    # Create tooltip label
    label = tk.Label(tw, text=text, justify='left', background="#ffffe0", relief='solid', borderwidth=1, font=("tahoma", "8", "normal"))
    label.pack(ipadx=1)
    
# Create a new window
window = tk.Tk()

# Set the window title
window.title("My GUI")

# Set the window size
window.geometry("500x500")

# Create input boxes and labels
input_label1 = tk.Label(window, text="Division:")
input_box1 = tk.Entry(window)
input_box1.insert(0, 'Enter division')
input_box1.bind("<FocusIn>", lambda event: input_box1.delete('0', 'end'))
input_label2 = tk.Label(window, text="URL:")
input_box2 = tk.Entry(window)
input_box2.insert(0, 'Enter URL')
input_box2.bind("<FocusIn>", lambda event: input_box2.delete('0', 'end'))
input_label3 = tk.Label(window, text="Output path:")
input_box3 = tk.Entry(window)
input_box3.insert(0, 'Enter Output path')
input_box3.bind("<FocusIn>", lambda event: input_box3.delete('0', 'end'))
input_label4 = tk.Label(window, text="Mode:")
input_box4 = tk.Entry(window)
input_box4.insert(0, 'Enter Mode')
input_box4.bind("<FocusIn>", lambda event: input_box4.delete('0', 'end'))

# Create button
button = tk.Button(window, text="Run Prediction", command=prediction_button_click)
button1 = tk.Button(window, text="scrape", command=data_scrape)

# Create output box
output_box = scrolledtext.ScrolledText(window, state=tk.DISABLED, height=20)

# Pack widgets in a grid
input_label1.grid(row=0, column=0)
input_box1.grid(row=1, column=0)
input_label2.grid(row=2, column=0)
input_box2.grid(row=3, column=0)
input_label3.grid(row=4, column=0)
input_box3.grid(row=5, column=0)
input_label4.grid(row=6, column=0)
input_box4.grid(row=7, column=0)
button.grid(row=8, column=0)
button1.grid(row=9, column=0)
output_box.grid(row=10, column=0, pady=10, padx=10)

# Add tooltip for Mode label
input_label4.bind("<Enter>", lambda event: show_tooltip(input_label4, "0: past data\n1: current matches\n2: last week match\n3: every upcoming match\n4: specific week"))
input_label4.bind("<Leave>", lambda event: input_label4.unbind("<Motion>"))

# Start the event loop
window.mainloop()