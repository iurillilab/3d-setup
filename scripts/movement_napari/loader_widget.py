import logging
from pathlib import Path

from napari.viewer import Viewer
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QWidget,
)

from movement.io import load_poses
from movement.napari.convert import ds_to_napari_tracks
from movement.napari.layer_styles import PointsStyle, TracksStyle
from movement.napari.utils import (
    columns_to_categorical_codes,
    set_playback_fps,
)

logger = logging.getLogger(__name__)


class Loader(QWidget):
    """Widget for loading data from files."""

    file_suffix_map = {
        "DeepLabCut": "Files containing predicted poses (*.h5 *.csv)",
        "LightningPose": "Files containing predicted poses (*.csv)",
        "SLEAP": "Files containing predicted poses (*.h5 *.slp)",
    }

    def __init__(self, napari_viewer: Viewer, parent=None):
        super().__init__(parent=parent)
        self.viewer = napari_viewer
        self.setLayout(QFormLayout())
        # Create widgets
        self.create_source_software_widget()
        self.create_fps_widget()
        self.create_file_path_widget()
        self.create_load_button()

    def create_source_software_widget(self):
        """Create a combo box for selecting the source software."""
        self.source_software_combo = QComboBox()
        self.source_software_combo.addItems(
            ["SLEAP", "DeepLabCut", "LightningPose"]
        )
        self.layout().addRow("source software:", self.source_software_combo)

    def create_fps_widget(self):
        """Create a spinbox for selecting the frames per second (fps)."""
        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setMinimum(1)
        self.fps_spinbox.setMaximum(1000)
        self.fps_spinbox.setValue(50)
        self.layout().addRow("fps:", self.fps_spinbox)

    def create_file_path_widget(self):
        """Create a line edit and browse button for selecting the file path.
        This allows the user to either browse the file system,
        or type the path directly into the line edit.
        """
        # File path line edit and browse button
        self.file_path_edit = QLineEdit()
        self.browse_button = QPushButton("browse")
        self.browse_button.clicked.connect(self.open_file_dialog)
        # Layout for line edit and button
        self.file_path_layout = QHBoxLayout()
        self.file_path_layout.addWidget(self.file_path_edit)
        self.file_path_layout.addWidget(self.browse_button)
        self.layout().addRow("pose file:", self.file_path_layout)

    def create_load_button(self):
        """Create a button to load the file and add layers to the viewer."""
        self.load_button = QPushButton("Load")
        self.load_button.clicked.connect(lambda: self.load_file())
        self.layout().addRow(self.load_button)

    def open_file_dialog(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.ExistingFile)
        # Allowed file suffixes based on the source software
        dlg.setNameFilter(
            self.file_suffix_map[self.source_software_combo.currentText()]
        )
        if dlg.exec_():
            file_paths = dlg.selectedFiles()
            # Set the file path in the line edit
            self.file_path_edit.setText(file_paths[0])

    def load_file(self):
        fps = self.fps_spinbox.value()
        source_software = self.source_software_combo.currentText()
        file_path = self.file_path_edit.text()
        if file_path == "":
            logger.warning("No file path specified.")
            return
        ds = load_poses.from_file(file_path, source_software, fps)

        self.data, self.props = ds_to_napari_tracks(ds)
        logger.info("Converted pose tracks to a napari Tracks array.")
        logger.debug(f"Tracks data shape: {self.data.shape}")

        self.file_name = Path(file_path).name
        self.add_layers()

        set_playback_fps(fps)
        logger.debug(f"Set napari playback speed to {fps} fps.")

    def add_layers(self):
        """Add the predicted pose tracks and keypoints to the napari viewer."""
        n_individuals = len(self.props["individual"].unique())
        color_by = "individual" if n_individuals > 1 else "keypoint"

        # Style properties for the napari Points layer
        points_style = PointsStyle(
            name=f"Keypoints - {self.file_name}",
            properties=self.props,
        )
        points_style.set_color_by(prop=color_by, cmap="turbo")

        # Track properties must be numeric, so convert str to categorical codes
        tracks_props = columns_to_categorical_codes(
            self.props, ["individual", "keypoint"]
        )

        # kwargs for the napari Tracks layer
        tracks_style = TracksStyle(
            name=f"Tracks - {self.file_name}",
            properties=tracks_props,
        )
        tracks_style.set_color_by(prop=color_by, cmap="turbo")

        # Add the new layers to the napari viewer
        self.viewer.add_tracks(self.data, **tracks_style.as_kwargs())
        self.viewer.add_points(self.data[:, 1:], **points_style.as_kwargs())
