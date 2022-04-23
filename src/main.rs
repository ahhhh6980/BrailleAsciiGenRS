use std::{
    collections::HashMap,
    error::Error,
    fs,
    io::{self, Write},
    ops::Range,
    path::{Path, PathBuf},
    slice::{ChunksExact, ChunksExactMut},
};

use image::{imageops, ImageBuffer, Rgb, Rgba};
mod fft2;
use fft2::*;

#[allow(dead_code)]
enum FindType {
    File,
    Dir,
}

fn list_dir<P: AsRef<Path>>(dir: P, find_dirs: FindType) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let mut files = Vec::<PathBuf>::new();
    for item in fs::read_dir(dir)? {
        let item = item?;
        match &find_dirs {
            FindType::File => {
                if item.file_type()?.is_file() {
                    files.push(item.path());
                }
            }
            FindType::Dir => {
                if item.file_type()?.is_dir() {
                    files.push(item.path());
                }
            }
        }
    }
    Ok(files)
}

fn prompt_number(bounds: Range<u32>, message: &str, def: i32) -> Result<u32, Box<dyn Error>> {
    let stdin = io::stdin();
    let mut buffer = String::new();
    // Tell the user to enter a value within the bounds
    if message != "" {
        if def >= 0 {
            println!(
                "{} in the range [{}:{}] (default: {})",
                message,
                bounds.start,
                bounds.end - 1,
                def
            );
        } else {
            println!(
                "{} in the range [{}:{}]",
                message,
                bounds.start,
                bounds.end - 1
            );
        }
    }
    buffer.clear();
    // Keep prompting until the user passes a value within the bounds
    Ok(loop {
        stdin.read_line(&mut buffer)?;
        print!("\r\u{8}");
        io::stdout().flush().unwrap();
        if let Ok(value) = buffer.trim().parse() {
            if bounds.contains(&value) {
                break value;
            }
        } else if def >= 0 {
            print!("\r\u{8}");
            print!("{}\n", &def);
            io::stdout().flush().unwrap();
            break def as u32;
        }
        buffer.clear();
    })
}

fn input_prompt<P: AsRef<Path>>(
    dir: P,
    find_dirs: FindType,
    message: &str,
) -> Result<PathBuf, Box<dyn Error>> {
    // Get files/dirs in dir
    let files = list_dir(&dir, find_dirs)?;
    // Inform the user that they will need to enter a value
    if message != "" {
        println!("{}", message);
    }
    // Enumerate the names of the files/dirs
    for (i, e) in files.iter().enumerate() {
        println!("{}: {}", i, e.display());
    }
    // This is the range of values they can pick
    let bound: Range<u32> = Range {
        start: 0,
        end: files.len() as u32,
    };
    // Return the path they picked
    Ok((&files[prompt_number(bound, "", -1)? as usize]).clone())
}

fn over_image(image: &ImageBuffer<Rgb<u8>, Vec<u8>>, w: usize, h: usize) -> String {
    let pixels = image;
    let rh: Vec<usize> = (0..h).into_iter().map(|x| x).collect();
    let rw: Vec<usize> = (0..w).into_iter().map(|x| x).collect();
    let chars = vec![
        "⠀", "⠁", "⠂", "⠃", "⠄", "⠅", "⠆", "⠇", "⠈", "⠉", "⠊", "⠋", "⠌", "⠍", "⠎", "⠏", "⠐", "⠑",
        "⠒", "⠓", "⠔", "⠕", "⠖", "⠗", "⠘", "⠙", "⠚", "⠛", "⠜", "⠝", "⠞", "⠟", "⠠", "⠡", "⠢", "⠣",
        "⠤", "⠥", "⠦", "⠧", "⠨", "⠩", "⠪", "⠫", "⠬", "⠭", "⠮", "⠯", "⠰", "⠱", "⠲", "⠳", "⠴", "⠵",
        "⠶", "⠷", "⠸", "⠹", "⠺", "⠻", "⠼", "⠽", "⠾", "⠿",
    ];
    let mut img = String::from("");
    for y in rh.chunks_exact(3) {
        for x in rw.chunks_exact(2) {
            let mut buf = 0;
            let mut i = 0;
            for height in y {
                for width in x {
                    i += 1;
                    let pix: Vec<u8> = image
                        .get_pixel(*width as u32, *height as u32)
                        .0
                        .try_into()
                        .unwrap();
                    let v = ((((pix.iter().fold(0, |x, y| x + y)) / 3) as f32) / 256.0).powf(2.0);
                    buf += if v > 0.00025 { 1 } else { 0 } * ((2.0f32.powf((i - 1) as f32)) as u8);
                }
            }
            img.push_str(chars[buf as usize]);
        }
        img.push_str("\n");
    }
    String::from(img)
    // let cols
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut file = input_prompt("input", FindType::File, "Please choose a media")?;
    let mut img = image::open(file)?;

    let size = prompt_number(0..2048, "Please pick a scale! (NxN):", 32)?;
    img = img.resize(size as u32, size as u32, imageops::FilterType::Gaussian);
    let pixels = img.to_rgb8();
    let (w, h) = (img.width(), img.height());

    let data: Vec<f32> = pixels.into_vec().iter().map(|x| *x as f32).collect();
    let kernel: Vec<f32> = vec![0.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 0.0];
    let mut kernel = FFT2Interface::from_vec(&kernel, (3, 3), 1.0);
    let mut fft_thing = FFT2Interface::from_vec(&data, (w as usize, h as usize), 1.0);

    fft_thing.convolve(&mut kernel);
    let gaussian_kernel: Vec<f32> = vec![1., 2., 1., 2., 4., 2., 1., 2., 1.];
    let mut gaussian_kernel = FFT2Interface::from_vec(&gaussian_kernel, (3, 3), 16.0);

    fft_thing.convolve(&mut gaussian_kernel);

    let (w, h) = fft_thing.size;
    let new_data = fft_thing.into_vec(256.0);
    let new_image = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_vec(
        w as u32,
        h as u32,
        new_data.iter().map(|x| *x as u8).collect(),
    )
    .unwrap();

    new_image.save("test.png");

    let txt = over_image(&new_image, w as usize, h as usize);
    println!("{}", txt);
    println!("Hello, world!");
    Ok(())
}
