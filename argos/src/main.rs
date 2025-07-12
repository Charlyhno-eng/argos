use opencv::{Result, core, highgui, prelude::*, videoio};

fn main() -> Result<()> {
    let window = "Cam Live";
    highgui::named_window(window, highgui::WINDOW_AUTOSIZE)?;

    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    if !videoio::VideoCapture::is_opened(&cam)? {
        panic!("Impossible d’ouvrir la caméra");
    }

    let mut frame = core::Mat::default();

    loop {
        cam.read(&mut frame)?;
        if frame.size()?.width == 0 {
            continue;
        }

        highgui::imshow(window, &frame)?;

        // Quitte si 'q' ou 'Esc' est pressé
        let key = highgui::wait_key(10)?;
        if key == 27 || key == 'q' as i32 {
            break;
        }
    }

    Ok(())
}
