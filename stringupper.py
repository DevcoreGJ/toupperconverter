import tkinter as tk

def convert_to_uppercase():
    input_str = entry.get()
    output_str = input_str.upper()
    output_label.config(text=output_str)

root = tk.Tk()
root.title("String Converter")

# Create widgets
entry_label = tk.Label(root, text="Enter a string:")
entry = tk.Entry(root)
convert_button = tk.Button(root, text="Convert", command=convert_to_uppercase)
output_label = tk.Label(root, text="")

# Add widgets to layout
entry_label.grid(row=0, column=0)
entry.grid(row=0, column=1)
convert_button.grid(row=1, column=0, columnspan=2)
output_label.grid(row=2, column=0, columnspan=2)

root.mainloop()
