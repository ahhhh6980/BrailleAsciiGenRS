use rayon::{iter::ParallelIterator, slice::ParallelSliceMut};
use rustdct::DctPlanner;
use rustfft::{num_complex::Complex, FftDirection, FftPlanner};
use transpose::transpose;

pub struct FFT2Interface {
    pub fft_planner: FftPlanner<f32>,
    pub dct_planner: DctPlanner<f32>,
    pub size: (usize, usize),
    pub data: Vec<Vec<Complex<f32>>>,
    pub channels: usize,
}

#[derive(Clone, Copy)]
enum Align {
    Center,
    Front,
    Back,
}

impl FFT2Interface {
    pub fn from_vec(data: &[f32], size: (usize, usize), divisor: f32) -> FFT2Interface {
        let channels = data.len() / (size.0 * size.1);
        let mut new_data = vec![vec![Complex::new(0.0, 0.0); 0]; channels];
        let mut j = 0;
        for item in data {
            j += 1;
            if j == channels {
                j = 0;
            }
            new_data[j].push(Complex {
                re: *item / divisor,
                im: 0.0,
            })
        }
        FFT2Interface {
            fft_planner: FftPlanner::new(),
            dct_planner: DctPlanner::new(),
            size,
            data: new_data,
            channels,
        }
    }
    pub fn fft2(&mut self, direction: FftDirection) {
        for i in 0..self.channels {
            let fft = self.fft_planner.plan_fft(self.size.0, direction);
            self.data[i]
                .par_chunks_exact_mut(self.size.0)
                .for_each_with(&fft, |fft, row| {
                    fft.process(row);
                });

            let mut new_data = vec![Complex::new(0.0, 0.0); self.size.0 * self.size.1];
            transpose::<Complex<f32>>(&self.data[i], &mut new_data, self.size.0, self.size.1);
            self.data[i] = new_data.clone();
            self.size = (self.size.1, self.size.0);

            let fft = self.fft_planner.plan_fft(self.size.0, direction);
            self.data[i]
                .par_chunks_exact_mut(self.size.0)
                .for_each_with(&fft, |fft, row| {
                    fft.process(row);
                });
        }
    }
    pub fn into_vec(self, scalar: f32) -> Vec<f32> {
        let mut out_vec = Vec::new();
        let (mut min, mut max) = (f32::MAX, f32::MIN);
        for channel in self.data.iter() {
            for e in channel {
                let v = e.norm_sqr();
                if v < min {
                    min = v;
                }
                if v > max {
                    max = v;
                }
            }
        }
        let (min, max) = (min.sqrt(), max.sqrt());
        for j in 0..self.size.0 * self.size.1 {
            for i in 0..self.channels {
                out_vec.push(
                    ((self.data[(i + 1) % self.channels][j].re - min) / (max - min)) * scalar,
                );
            }
        }
        out_vec
    }
    pub fn into_vec_copy_channels(self, channels: usize, scalar: f32) -> Vec<f32> {
        let mut out_vec = Vec::new();
        let (mut min, mut max) = (f32::MAX, f32::MIN);
        for i in 0..self.channels {
            for j in 0..(self.size.0 * self.size.1) {
                let v = self.data[i][j].norm_sqr();
                if v < min {
                    min = v;
                }
                if v > max {
                    max = v;
                }
            }
        }
        let (min, max) = (min.sqrt(), max.sqrt());
        for j in 0..self.size.0 * self.size.1 {
            for i in 0..(self.channels * channels) {
                out_vec.push(
                    ((self.data[(i + 1) % self.channels][j].re - min) / (max - min)) * scalar,
                );
            }
        }
        out_vec
    }
    pub fn pad(&mut self, x: (usize, usize), y: (usize, usize)) {
        let mut new_data =
            vec![vec![Complex::new(0.0, 0.0); (self.size.0 + x.0 + x.1) * y.0]; self.channels];

        for (channel, new_row) in self.data.iter().zip(&mut new_data) {
            for row in channel.chunks_exact(self.size.0) {
                new_row.extend(vec![Complex::new(0.0, 0.0); x.0].into_iter());
                new_row.extend_from_slice(row);
                new_row.extend(vec![Complex::new(0.0, 0.0); x.1].into_iter());
            }
        }
        for item in new_data.iter_mut() {
            item.extend(vec![
                Complex::new(0.0, 0.0);
                (self.size.0 + x.0 + x.1) * y.1
            ]);
        }
        self.data = new_data;
        self.fft_planner = FftPlanner::new();
        self.dct_planner = DctPlanner::new();
        self.size = (self.size.0 + x.0 + x.1, self.size.1 + y.0 + y.1);
    }
    pub fn pad_to_square(&mut self, mode: Align) -> ((usize, usize), Align) {
        let old_size = self.size;
        let size_diff = (self.size.0 as i32 - self.size.1 as i32).abs() as usize;
        let resize = match mode {
            Align::Center => (size_diff / 2, size_diff / 2),
            Align::Front => (0, size_diff),
            Align::Back => (size_diff, 0),
        };
        if self.size.0 > self.size.1 {
            self.pad((0, 0), resize);
        } else {
            self.pad(resize, (0, 0));
        }
        (old_size, mode)
    }
    pub fn remove_square_padding(&mut self, method: ((usize, usize), Align)) {
        let size_diff = ((method.0 .0 as i32) - (method.0 .1 as i32)).abs() as usize;
        let paddings = match method.1 {
            Align::Center => (size_diff / 2, size_diff / 2),
            Align::Front => (0, size_diff),
            Align::Back => (size_diff, 0),
        };
        let (mut x_range, mut y_range) = (0..method.0 .0, 0..method.0 .1);
        if method.0 .0 > method.0 .1 {
            y_range = paddings.0..((self.size.0 * self.size.1) - paddings.1);
        } else {
            x_range = paddings.0..(self.size.0 - paddings.1);
        }
        let mut new_data = vec![Vec::<Complex<f32>>::new(); 4];
        for (channel, new_channel) in self.data.iter_mut().zip(&mut new_data) {
            for row in channel[y_range.clone()].chunks_exact_mut(method.0 .0) {
                new_channel.extend(&row.to_vec()[x_range.clone()]);
            }
        }
        self.data = new_data;
        self.size = method.0;
        self.fft_planner = FftPlanner::new();
        self.dct_planner = DctPlanner::new();
    }
    pub fn crop(&mut self, offset: (usize, usize), size: (usize, usize)) {
        let (x_range, y_range) = (
            offset.0..(size.0 + offset.0),
            (offset.1 * self.size.0)..((offset.1 * self.size.0) + (self.size.0 * size.1)),
        );
        let mut new_data = vec![Vec::<Complex<f32>>::new(); 4];
        for (channel, new_channel) in self.data.iter_mut().zip(&mut new_data) {
            for row in channel[y_range.clone()].chunks_exact_mut(self.size.0) {
                new_channel.extend(&row.to_vec()[x_range.clone()]);
            }
        }
        self.data = new_data;
        self.size = size;
        self.fft_planner = FftPlanner::new();
        self.dct_planner = DctPlanner::new();
    }
    pub fn crop_align(&mut self, mode: (Align, Align), size: (usize, usize)) {
        let offset = (
            match mode.0 {
                Align::Center => (self.size.0 / 2) - (size.0 / 2),
                Align::Front => 0,
                Align::Back => self.size.0 - size.0,
            },
            match mode.1 {
                Align::Center => (self.size.1 / 2) - (size.1 / 2),
                Align::Front => 0,
                Align::Back => self.size.1 - size.1,
            },
        );
        //
        self.crop(offset, size);
    }
    #[allow(unused_variables)]
    pub fn convolve(&mut self, kernel: &mut FFT2Interface) {
        kernel.pad((0, self.size.0 + 3), (0, self.size.1 + 3));
        self.pad((3, 3), (3, 3));

        kernel.fft2(FftDirection::Forward);
        self.fft2(FftDirection::Forward);

        for channel in self.data.iter_mut() {
            for (var, kernel_var) in channel.iter_mut().zip(&kernel.data[0]) {
                *var *= kernel_var;
            }
        }

        kernel.fft2(FftDirection::Inverse);
        self.fft2(FftDirection::Inverse);
    }
    pub fn deconvolve(&mut self, kernel: &mut FFT2Interface) {
        let dirac: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let mut dirac = FFT2Interface::from_vec(&dirac, (3, 3), 1.0);

        dirac.pad((0, self.size.0 - 3), (0, self.size.1 - 3));

        dirac.fft2(FftDirection::Forward);
        kernel.fft2(FftDirection::Forward);
        self.fft2(FftDirection::Forward);

        let s = 1.0 / ((self.size.0) as f32 * (self.size.1) as f32);

        for channel in self.data.iter_mut() {
            for ((var, kernel_var), dirac_var) in
                channel.iter_mut().zip(&kernel.data[0]).zip(&dirac.data[0])
            {
                *var *= (dirac_var) / (kernel_var + s);
            }
        }

        kernel.fft2(FftDirection::Inverse);
        self.fft2(FftDirection::Inverse);
    }
}
