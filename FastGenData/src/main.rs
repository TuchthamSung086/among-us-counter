#![deny(unused_must_use)]

use std::{env::args, fmt::Write, fs, sync::Mutex, time::Instant};

use color_eyre::eyre::{Context, ContextCompat, Result};
use image::{imageops, DynamicImage, GrayAlphaImage, GrayImage, LumaA, Pixel, RgbaImage};
use imageproc::{
    definitions::Image,
    geometric_transformations::{self, Interpolation},
};
use indicatif::ParallelProgressIterator;
use itertools::Itertools;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

#[derive(Clone, Debug)]
struct PasteParams {
    sz: f32,
    rot: f32,
    flip: bool,
    mirror: bool,
    x: f32,
    y: f32,
}

fn paste_on<I>(fg: &Image<I>, bg: &mut Image<I>, params: &PasteParams)
where
    I: Pixel<Subpixel = u8> + Send + Sync + 'static,
{
    let mut fg = fg.clone();
    let (width, height) = bg.dimensions();
    let base_width = width as f32 / 10.;

    let (fw, fh) = fg.dimensions();
    if params.flip {
        imageops::flip_vertical_in_place(&mut fg)
    }
    if params.mirror {
        imageops::flip_horizontal_in_place(&mut fg)
    }

    fg = imageops::resize(
        &fg,
        (base_width * params.sz) as u32,
        ((base_width * params.sz / fw as f32) * fh as f32) as u32,
        imageops::FilterType::CatmullRom,
    );

    let zeroed = *I::from_slice((vec![0u8; I::CHANNEL_COUNT as usize]).as_slice());
    fg = geometric_transformations::rotate_about_center(
        &fg,
        params.rot,
        Interpolation::Nearest,
        zeroed,
    );

    let loc_x = (params.x * width as f32) as _;
    let loc_y = (params.y * height as f32) as _;

    imageops::overlay(bg, &fg, loc_x, loc_y);
}

fn random_paste(rng: &mut impl rand::Rng) -> PasteParams {
    PasteParams {
        sz: rng.gen_range(1.0..=2.0),
        rot: rng.gen_range(0.0..360.0),
        flip: rng.gen_bool(0.5),
        mirror: rng.gen_bool(0.5),
        x: rng.gen_range(0.0..=1.0),
        y: rng.gen_range(0.0..=1.0),
    }
}

fn is_img_overlap(img1: &[u8], img2: &[u8]) -> bool {
    let dic1 = img1.iter().copied().counts();
    let dic2 = img2.iter().copied().counts();
    for &key in dic1.keys() {
        let d1 = dic1[&key];
        let d2 = *dic2.get(&key).unwrap_or(&0);
        if key != 0 && d1 > d2 {
            return true;
        }
    }
    return false;
}

fn is_in_boundary(fg: &GrayAlphaImage, bg: &GrayAlphaImage, params: &PasteParams) -> bool {
    // Honestly, i have no ideae what is going on here.

    let (width, height) = bg.dimensions();
    let (fwidth, fheight) = fg.dimensions();
    let cx = width / 2;
    let cy = height / 2;
    let theta = params.rot;

    let rad = theta.to_radians();
    let base_width = width / 10;
    let fg = imageops::resize(
        fg,
        (params.sz * base_width as f32) as u32,
        (base_width as f32 / fwidth as f32 * fheight as f32 * params.sz) as u32,
        imageops::FilterType::Nearest,
    );
    let (fwidth, fheight) = fg.dimensions();
    let (fwidth, fheight) = (fwidth as i32, fheight as i32);
    let loc_x = (params.x * width as f32) as i32;
    let loc_y = (params.y * height as f32) as i32;
    let corners = [
        [loc_x, loc_y],
        [loc_x + fwidth, loc_y],
        [loc_x, loc_y + fheight],
        [loc_x + fwidth, loc_y + fheight],
    ];
    let (cx, cy) = (cx as i32, cy as i32);
    for [px, py] in corners.into_iter() {
        let new_px = cx + ((px - cx) as f32 * rad.cos() + (py - cy) as f32 * rad.sin()) as i32;
        let new_py = cy + (-((px - cx) as f32 * rad.sin()) + (py - cy) as f32 * rad.cos()) as i32;
        if new_px > width as i32 || new_px < 0 {
            return false;
        }
        if new_py > height as i32 || new_py < 0 {
            return false;
        }
    }
    return true;
}

fn gen(fg: &RgbaImage, bg: RgbaImage, it: usize) -> (RgbaImage, GrayImage, usize) {
    let (width, height) = bg.dimensions();
    let mut label = GrayAlphaImage::new(width, height);
    let mut img = bg;
    let mut cnt = 0;
    let mut overlap_cnt = 0;
    let mut boundary_cnt = 0;

    let rng = &mut rand::thread_rng();

    for i in 0..it {
        let paste_params = random_paste(rng);
        let figure_img = imageproc::map::map_colors(fg, |rgba| {
            let alpha = rgba[3];
            LumaA([255 - (255 / it * i) as u8, alpha])
        });
        if !is_in_boundary(&figure_img, &label, &paste_params) {
            boundary_cnt += 1;
            continue;
        }

        let mut label_new = label.clone();
        paste_on(&figure_img, &mut label_new, &paste_params);

        if is_img_overlap(&label, &label_new) {
            overlap_cnt += 1;
            continue;
        }

        paste_on(fg, &mut img, &paste_params);

        label = label_new;
        cnt += 1;
    }

    // println!("boundary_cnt={boundary_cnt}, overlap={overlap_cnt}");

    let label = DynamicImage::from(label).to_luma8();
    (img, label, cnt)
}

fn main() -> Result<()> {
    color_eyre::install()?;

    // Read input images
    let canteen: RgbaImage = image::open("./bg/canteen.jpg")?.into_rgba8();
    let red: RgbaImage = image::open("./crewmate/red.png")?.into_rgba8();

    // Number of generated image, recived from command line argument
    let n = args()
        .skip(1)
        .next()
        .context("n is missing (args[1] is missing)")
        .and_then(|s| s.parse().context("n cannot be convert to usize"))
        .context("Fail parsing {n}, (usage: cargo run --release <n>)")?;

    // Create output folder, if not exist
    fs::create_dir_all("results")?;

    // Begin the processing
    let start = Instant::now();
    {
        // Vector to store the count in each image
        // Each element is (image_id, count)
        let count_storage = Mutex::new(Vec::with_capacity(n));

        // Run processing task in parallel
        (0..n)
            .into_par_iter()
            .progress_count(n as u64)
            .try_for_each(|i| {
                let (img, label, cnt): (RgbaImage, GrayImage, usize) =
                    gen(&red, canteen.clone(), 10);
                // println!("i={i}, count={cnt}");
                img.save(format!("results/img-{i}.png"))?;
                label.save(format!("results/label-{i}.png"))?;
                count_storage.lock().unwrap().push((i, cnt));

                Result::<(), color_eyre::Report>::Ok(())
            })?;

        // Sort the count vector by image id
        // This is "need" because task are run in parallel, so image id's ordering is not necessary ascending.
        let mut count_storage = count_storage.into_inner()?;
        count_storage.sort_unstable();

        // Write data from count vector to file
        let mut contents = String::new();
        for (i, cnt) in count_storage {
            writeln!(contents, "{i},{cnt}")?;
        }
        fs::write("results/count.txt", contents)?;
    }
    let end = Instant::now();
    let duration = end - start;
    println!("================================================================");
    println!("take {:?} for {} image", duration, n);
    println!("average cycle time = {:?}", duration / n as _);
    println!("throughput = {} item/s", n as f64 / duration.as_secs_f64());

    Ok(())
}
