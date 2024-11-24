import tkinter as tk
from tkinter import scrolledtext
from my_neural_network import TextGenerator

# Инициализация генератора текста
text_generator = TextGenerator()

class ChatApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("Chat with AI")
        self.root.geometry("400x500")
        
        # Создание области для сообщений
        self.chat_area = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, bg="#2C2F33", fg="#FFFFFF", font=("Arial", 12))
        self.chat_area.pack(pady=10, padx=10)
        self.chat_area.config(state=tk.DISABLED)

        # Создание поля ввода сообщения
        self.entry_frame = tk.Frame(self.root, bg="#23272A")
        self.entry_frame.pack(pady=5, padx=10)
        
        self.entry_field = tk.Entry(self.entry_frame, bg="#2C2F33", fg="#FFFFFF", font=("Arial", 12), width=30)
        self.entry_field.pack(side=tk.LEFT, pady=5, padx=5)
        self.entry_field.bind("<Return>", self.send_message)

        self.send_button = tk.Button(self.entry_frame, text="Отправить", bg="#7289DA", fg="#FFFFFF", command=self.send_message, font=("Arial", 12))
        self.send_button.pack(side=tk.RIGHT, pady=5, padx=5)

    def send_message(self, event=None):
        user_input = self.entry_field.get()
        if user_input.strip() != "":
            self.display_message(f"Вы: {user_input}\n")
            self.entry_field.delete(0, tk.END)
            
            response = text_generator.generate_answer(user_input)
            self.display_message(f"ИИ: {response}\n")

    def display_message(self, message):
        self.chat_area.config(state=tk.NORMAL)
        self.chat_area.insert(tk.END, message)
        self.chat_area.yview(tk.END)
        self.chat_area.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatApplication(root)
    root.mainloop()
