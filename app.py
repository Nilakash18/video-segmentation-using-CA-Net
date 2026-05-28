import streamlit as st
import os

from utils import (
    process_video,
    OUTPUT_SEG_VIDEO,
    OUTPUT_BLEND_VIDEO,
    OUTPUT_SIDE_VIDEO
)

# ===== Streamlit Page =====

st.set_page_config(
    page_title="CA-Net Video Segmentation",
    layout="wide"
)

st.title("🎥 CA-Net Video Segmentation App")

st.write(
    "Upload a video and run CA-DeepLab segmentation."
)

# ===== Upload =====

uploaded_file = st.file_uploader(
    "Upload Video",
    type=["mp4","avi","mov"]
)

if uploaded_file is not None:

    save_path = os.path.join(
        "uploads",
        uploaded_file.name
    )

    with open(save_path, "wb") as f:
        f.write(
            uploaded_file.getbuffer()
        )

    st.success(
        "Video uploaded successfully!"
    )

    if st.button(
        "Run Segmentation"
    ):

        with st.spinner(
            "Processing video..."
        ):
            process_video(
                save_path,
                resize_output=(640, 360),
                frame_skip=2,
                max_frames=300
            )

        st.success(
            "Segmentation Complete!"
        )

        # ===== Show Outputs =====

        st.subheader(
            "Segmented Output"
        )

        st.video(
            OUTPUT_SEG_VIDEO
        )

        st.subheader(
            "Blend Output"
        )

        st.video(
            OUTPUT_BLEND_VIDEO
        )

        st.subheader(
            "Side-by-Side Output"
        )

        st.video(
            OUTPUT_SIDE_VIDEO
        )

        # ===== Downloads =====

        st.subheader(
            "Download Outputs"
        )

        with open(
            OUTPUT_SEG_VIDEO,
            "rb"
        ) as f:

            st.download_button(
                "Download Segmentation",
                f,
                file_name="seg.mp4"
            )

        with open(
            OUTPUT_BLEND_VIDEO,
            "rb"
        ) as f:

            st.download_button(
                "Download Blend Video",
                f,
                file_name="blend.mp4"
            )

        with open(
            OUTPUT_SIDE_VIDEO,
            "rb"
        ) as f:

            st.download_button(
                "Download Side Video",
                f,
                file_name="side.mp4"
            )