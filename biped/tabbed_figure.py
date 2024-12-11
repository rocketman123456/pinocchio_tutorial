from __future__ import annotations

import sys
import signal
import typing as tp
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.qt_compat import QtCore, QtWidgets, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT as NavigationToolbar

from qt_utils import get_qt_app


## @brief Class representing a tab holding a figure.
class FigTab(QtWidgets.QWidget):
    ## @brief Constructor.
    ## @param  parent   Parent widget.
    ## @param  fig      matplotlib figure to be inserted into the tab.
    def __init__(self, parent, fig):
        super(FigTab, self).__init__(parent)

        self.parent = parent
        self.wdg = QtWidgets.QWidget()

        self.canvas = FigureCanvas(fig)
        self.toolbar = NavigationToolbar(self.canvas, self.parent)

        self.layout = QtWidgets.QVBoxLayout(self.wdg)
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)

        self.setLayout(self.layout)


## @brief A windows with several tabs holding matplotlib figures.
class TabbedFigure(QtWidgets.QMainWindow):

    ## brief Constructor.
    def __init__(self):
        # create an app if it does not exist
        TabbedFigure.get_qt_app()

        # init
        super().__init__()

        self.setWindowIcon(QtGui.QIcon(str(Path(__file__).parent / "icon.png")))

        # timer to allow Python interpreter to handel signals periodically
        self.timer = QtCore.QTimer()

        # Create layout.
        self.wdg = QtWidgets.QWidget()
        self.tab_wdg = QtWidgets.QTabWidget(self)
        self.layout = QtWidgets.QVBoxLayout(self.wdg)
        self.layout.addWidget(self.tab_wdg)
        self.setCentralWidget(self.wdg)

        self.subtabs = {}

        signal.signal(signal.SIGINT, self.sigint_handler)

    @staticmethod
    def sigint_handler(sig, frame):
        """signal handler. Called when 'Ctrl-C' and SIGINT are sent"""
        print("Got SIGINT, exiting.")
        TabbedFigure.get_qt_app().closeAllWindows()
        sys.exit(0)

    def setup_timer(self):
        """setup a timer to wake the python interpreter"""

        def tick():
            pass

        self.timer.start(500)  # You may change this if you wish.
        self.timer.timeout.connect(tick)  # Let the interpreter run each 500 ms.

    ## @brief Create a Qt application. Must be called before creating a TabbedFigure.
    @staticmethod
    def get_qt_app():
        return get_qt_app()

    ## @brief Executes a Qt application. Must be called after the TabbedFigure has been built.
    @staticmethod
    def exec_qt_app(qt_app: tp.Optional[QtWidgets.QApplication] = None) -> tp.NoReturn:
        if qt_app is None:
            qt_app = TabbedFigure.get_qt_app()
        try:
            qt_app.exec_()
        except:
            pass

    def add_figure(self, name, fig, tab_name=None):
        """
        Create a new tab an insert a figure inside it.
        Parameters:
            - name: String appearing in the tab thumbnail.
            - fig: matplotlib figure to be inserted into the tab.
            - tab: If set, name of the tab where to place the figure.
                   If None, the tab is added to the main window.
        """
        if tab_name is None:
            self.tab_wdg.addTab(FigTab(self, fig), name)
        else:
            self.subtabs[tab_name].addTab(FigTab(self, fig), name)

    def add_tab(self, tab_name):
        """
        Add a tab to the window - itself cabable of including figures.
        Parameters:
            - tab_name: Name of the new tab.
        """
        self.subtabs[tab_name] = QtWidgets.QTabWidget(self.tab_wdg)
        layout = QtWidgets.QVBoxLayout(self.wdg)
        layout.addWidget(self.subtabs[tab_name])
        self.tab_wdg.addTab(self.subtabs[tab_name], tab_name)

    ## @brief Set the string appeating in the window title bar.
    ## @param title
    def set_window_title(self, title):
        self.setWindowTitle(title)

    ## @brief Display the window. Must be called once after all the tabs have been created.
    def show(self):
        self.setup_timer()
        self.showMaximized()
