import tkinter as tk 
from tkinter import ttk 
from nn import train_and_visualize

def run_function():
    p = problem_types_var.get()
    o = optimizers_var.get()
    train_and_visualize(p, o)
    label_result.config(text=f"NN ran with {p} and {o}")


problem_types = ['classification', 'regression']
optimizers = ['sgd', 'momentum', 'rmsprop', 'adam', 'adagrad']

root = tk.Tk()
root.title('Neural Network')

problem_types_var = tk.StringVar(value=problem_types[0])
optimizers_var = tk.StringVar(value=optimizers[0])

dropdown_pt = ttk.Combobox(root, textvariable=problem_types_var, values=problem_types, state='readonly') 
dropdown_pt.pack(pady=10)
optimizer_pt = ttk.Combobox(root, textvariable=optimizers_var, values=optimizers, state='readonly') 
optimizer_pt.pack(pady=10)

button = tk.Button(root, text='Run', command=run_function)
button.pack(pady=10)

label_result = tk.Label(root, text="Select the options and click the button.")
label_result.pack(pady=10)

root.mainloop()
