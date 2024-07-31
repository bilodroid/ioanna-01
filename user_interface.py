from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QTabWidget, QTextEdit, QScrollArea
from PyQt5.QtGui import QImage, QPixmap
from conversation_module import ConversationModule

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.conversation_module = ConversationModule()
        self.camera = self.conversation_module.camera
        self.setWindowTitle("Ioanna-1")
        self.setGeometry(100, 100, 800, 600)
        self.create_widgets()
        self.conversation_module.new_message.connect(self.add_new_message)

    def create_widgets(self):
        main_layout = QVBoxLayout(self)
        
        self.tabs = QTabWidget()
        
        self.camera_tab = QWidget()
        camera_layout = QVBoxLayout(self.camera_tab)
        self.camera_widget = QLabel(self.camera_tab)
        self.camera_widget.setAlignment(Qt.AlignCenter)
        camera_layout.addWidget(self.camera_widget)
        self.camera_tab.setLayout(camera_layout)
        
        self.chat_tab = QWidget()
        chat_layout = QVBoxLayout(self.chat_tab)
        self.chat_text = QTextEdit(self.chat_tab)
        self.chat_text.setReadOnly(True)
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.chat_text)
        scroll_area.setWidgetResizable(True)
        chat_layout.addWidget(scroll_area)
        self.chat_tab.setLayout(chat_layout)
        
        self.tabs.addTab(self.camera_tab, "Camera")
        self.tabs.addTab(self.chat_tab, "Chat")
        
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)
        self.camera.start_camera()
        self.update_camera_feed()

    def update_camera_feed(self):
        frame = self.camera.get_current_frame()
        if frame is not None:
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_image)
            self.camera_widget.setPixmap(pixmap)
        QTimer.singleShot(30, self.update_camera_feed)

    @pyqtSlot(dict)
    def add_new_message(self, message):
        if message['role'] == 'assistant':
            self.chat_text.append(f"Ioanna: {message['content']}")
        elif message['role'] == 'user':
            self.chat_text.append(f"Me: {message['content']}")
        self.chat_text.append("")
        self.chat_text.verticalScrollBar().setValue(self.chat_text.verticalScrollBar().maximum())

    def run(self):
        self.show()
        self.conversation_module.start()
#
#
#
#
#
