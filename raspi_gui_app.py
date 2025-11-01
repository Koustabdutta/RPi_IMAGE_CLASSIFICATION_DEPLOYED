"""
Raspberry Pi Image Classifier GUI
User-friendly interface for image classification on Raspberry Pi 3B
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import time
import os

# Try to import TFLite runtime (lightweight) or fall back to TensorFlow
try:
    import tflite_runtime.interpreter as tflite
    print("Using TFLite Runtime")
except ImportError:
    import tensorflow as tf
    tflite = tf.lite
    print("Using TensorFlow Lite")

class ImageClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🖼️ Image Classifier - Raspberry Pi 3B")
        self.root.geometry("800x600")
        self.root.configure(bg='#2c3e50')
        
        # Model parameters
        self.model_path = 'image_classifier_quantized.tflite'
        self.class_names_path = 'class_names.txt'
        self.img_size = 32
        
        # State variables
        self.interpreter = None
        self.class_names = []
        self.current_image = None
        self.current_image_path = None
        
        # Setup UI
        self.setup_ui()
        
        # Load model and class names
        self.load_model()
        self.load_class_names()
    
    def setup_ui(self):
        """Setup the user interface"""
        
        # Title
        title_label = tk.Label(
            self.root,
            text="🎯 Image Classification System",
            font=('Arial', 20, 'bold'),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        title_label.pack(pady=20)
        
        # Status label
        self.status_label = tk.Label(
            self.root,
            text="Ready to classify images",
            font=('Arial', 10),
            bg='#2c3e50',
            fg='#95a5a6'
        )
        self.status_label.pack()
        
        # Main container
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)
        
        # Left panel - Image display
        left_panel = tk.Frame(main_frame, bg='#34495e', relief=tk.RAISED, borderwidth=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        tk.Label(
            left_panel,
            text="Image Preview",
            font=('Arial', 12, 'bold'),
            bg='#34495e',
            fg='#ecf0f1'
        ).pack(pady=10)
        
        # Image canvas
        self.image_label = tk.Label(left_panel, bg='#34495e')
        self.image_label.pack(pady=20, padx=20)
        
        # Right panel - Results
        right_panel = tk.Frame(main_frame, bg='#34495e', relief=tk.RAISED, borderwidth=2)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        tk.Label(
            right_panel,
            text="Classification Results",
            font=('Arial', 12, 'bold'),
            bg='#34495e',
            fg='#ecf0f1'
        ).pack(pady=10)
        
        # Results text area
        self.results_text = tk.Text(
            right_panel,
            font=('Courier', 11),
            bg='#2c3e50',
            fg='#ecf0f1',
            height=15,
            width=35,
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        self.results_text.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        # Button frame
        button_frame = tk.Frame(self.root, bg='#2c3e50')
        button_frame.pack(pady=20)
        
        # Buttons
        btn_style = {
            'font': ('Arial', 11, 'bold'),
            'bg': '#3498db',
            'fg': 'white',
            'activebackground': '#2980b9',
            'activeforeground': 'white',
            'relief': tk.RAISED,
            'borderwidth': 2,
            'padx': 20,
            'pady': 10,
            'cursor': 'hand2'
        }
        
        self.load_btn = tk.Button(
            button_frame,
            text="📁 Load Image",
            command=self.load_image,
            **btn_style
        )
        self.load_btn.pack(side=tk.LEFT, padx=5)
        
        self.classify_btn = tk.Button(
            button_frame,
            text="🔍 Classify Image",
            command=self.classify_image,
            state=tk.DISABLED,
            **btn_style
        )
        self.classify_btn.pack(side=tk.LEFT, padx=5)
        
        # Try to add camera button if picamera is available
        try:
            import picamera
            self.camera_btn = tk.Button(
                button_frame,
                text="📷 Capture Photo",
                command=self.capture_photo,
                **btn_style
            )
            self.camera_btn.pack(side=tk.LEFT, padx=5)
        except ImportError:
            pass
        
        self.clear_btn = tk.Button(
            button_frame,
            text="🗑️ Clear",
            command=self.clear_results,
            **btn_style
        )
        self.clear_btn.pack(side=tk.LEFT, padx=5)
    
    def load_model(self):
        """Load TFLite model"""
        try:
            if not os.path.exists(self.model_path):
                messagebox.showerror(
                    "Error",
                    f"Model file not found: {self.model_path}\n"
                    "Please ensure the quantized model is in the same directory."
                )
                return
            
            self.update_status("Loading model...")
            self.interpreter = tflite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            self.update_status("✓ Model loaded successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            self.update_status("❌ Model loading failed")
    
    def load_class_names(self):
        """Load class names from file"""
        try:
            if not os.path.exists(self.class_names_path):
                messagebox.showwarning(
                    "Warning",
                    f"Class names file not found: {self.class_names_path}\n"
                    "Using default class indices."
                )
                self.class_names = [f"Class {i}" for i in range(10)]
                return
            
            with open(self.class_names_path, 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]
            
            self.update_status(f"✓ Loaded {len(self.class_names)} class names")
            
        except Exception as e:
            messagebox.showwarning("Warning", f"Failed to load class names:\n{str(e)}")
            self.class_names = [f"Class {i}" for i in range(10)]
    
    def load_image(self):
        """Load image from file"""
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Load and display image
                self.current_image_path = file_path
                image = Image.open(file_path)
                self.current_image = image.copy()
                
                # Display image
                display_image = image.copy()
                display_image.thumbnail((300, 300), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(display_image)
                
                self.image_label.configure(image=photo)
                self.image_label.image = photo
                
                # Enable classify button
                self.classify_btn.configure(state=tk.NORMAL)
                
                self.update_status(f"✓ Image loaded: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
    
    def capture_photo(self):
        """Capture photo from Pi Camera"""
        try:
            from picamera import PiCamera
            import io
            
            self.update_status("📷 Capturing photo...")
            
            # Initialize camera
            camera = PiCamera()
            camera.resolution = (640, 480)
            time.sleep(2)  # Camera warm-up
            
            # Capture to memory
            stream = io.BytesIO()
            camera.capture(stream, format='jpeg')
            camera.close()
            
            # Load image
            stream.seek(0)
            image = Image.open(stream)
            self.current_image = image.copy()
            
            # Display image
            display_image = image.copy()
            display_image.thumbnail((300, 300), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(display_image)
            
            self.image_label.configure(image=photo)
            self.image_label.image = photo
            
            # Enable classify button
            self.classify_btn.configure(state=tk.NORMAL)
            
            self.update_status("✓ Photo captured successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to capture photo:\n{str(e)}")
    
    def preprocess_image(self, image):
        """Preprocess image for model input"""
        # Resize to model input size
        image = image.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Convert to uint8 (for quantized model)
        img_array = img_array.astype(np.uint8)
        
        return img_array
    
    def classify_image(self):
        """Classify the loaded image"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
        
        if self.interpreter is None:
            messagebox.showerror("Error", "Model not loaded!")
            return
        
        try:
            self.update_status("🔍 Classifying image...")
            
            # Preprocess image
            input_data = self.preprocess_image(self.current_image)
            
            # Run inference
            start_time = time.time()
            
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Get predictions
            predictions = output[0]
            
            # Handle uint8 output (scale back to 0-1 range)
            if predictions.dtype == np.uint8:
                predictions = predictions.astype(np.float32) / 255.0
            
            # Get top 3 predictions
            top_indices = np.argsort(predictions)[-3:][::-1]
            
            # Display results
            self.display_results(predictions, top_indices, inference_time)
            
            self.update_status(f"✓ Classification complete ({inference_time:.1f}ms)")
            
        except Exception as e:
            messagebox.showerror("Error", f"Classification failed:\n{str(e)}")
            self.update_status("❌ Classification failed")
    
    def display_results(self, predictions, top_indices, inference_time):
        """Display classification results"""
        self.results_text.delete(1.0, tk.END)
        
        # Header
        self.results_text.insert(tk.END, "=" * 40 + "\n")
        self.results_text.insert(tk.END, "  CLASSIFICATION RESULTS\n")
        self.results_text.insert(tk.END, "=" * 40 + "\n\n")
        
        # Top predictions
        self.results_text.insert(tk.END, "Top 3 Predictions:\n\n")
        
        for i, idx in enumerate(top_indices, 1):
            class_name = self.class_names[idx] if idx < len(self.class_names) else f"Class {idx}"
            confidence = predictions[idx] * 100
            
            self.results_text.insert(tk.END, f"{i}. {class_name}\n")
            self.results_text.insert(tk.END, f"   Confidence: {confidence:.2f}%\n")
            
            # Draw confidence bar
            bar_length = int(confidence / 3)
            bar = "█" * bar_length + "░" * (33 - bar_length)
            self.results_text.insert(tk.END, f"   [{bar}]\n\n")
        
        # Performance info
        self.results_text.insert(tk.END, "=" * 40 + "\n")
        self.results_text.insert(tk.END, f"Inference Time: {inference_time:.1f} ms\n")
        self.results_text.insert(tk.END, f"FPS: {1000/inference_time:.1f}\n")
        self.results_text.insert(tk.END, "=" * 40 + "\n")
    
    def clear_results(self):
        """Clear current results"""
        self.current_image = None
        self.current_image_path = None
        self.image_label.configure(image='')
        self.image_label.image = None
        self.results_text.delete(1.0, tk.END)
        self.classify_btn.configure(state=tk.DISABLED)
        self.update_status("Ready to classify images")
    
    def update_status(self, message):
        """Update status label"""
        self.status_label.configure(text=message)
        self.root.update()

def main():
    """Main application entry point"""
    root = tk.Tk()
    app = ImageClassifierGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
