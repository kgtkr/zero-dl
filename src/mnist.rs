use byteorder::{BigEndian, ReadBytesExt};
use ndarray::prelude::*;
use std::convert::TryFrom;
use std::io::Read;

#[derive(Debug, Clone)]
pub struct MnistLabels {
    pub labels: Array1<u8>,
}

impl MnistLabels {
    pub fn parse(source: &mut impl Read) -> Option<MnistLabels> {
        source.read_i32::<BigEndian>().ok().filter(|&x| x == 2049)?;
        let len = source
            .read_i32::<BigEndian>()
            .ok()
            .and_then(|x| usize::try_from(x).ok())?;
        let mut data = Vec::with_capacity(len);
        source.read_to_end(&mut data).ok()?;

        let labels = Array::from_shape_vec((len,), data).ok()?;

        Some(MnistLabels { labels })
    }

    pub fn to_data(&self) -> Array2<f32> {
        let mut res = Array::zeros((self.labels.len(), 10));
        for i in 0..self.labels.len() {
            res[[i, self.labels[i] as usize]] = 1.;
        }
        res
    }
}

#[derive(Debug, Clone)]
pub struct MnistImages {
    pub width: usize,
    pub height: usize,
    pub images: Array3<u8>,
}

impl MnistImages {
    pub fn parse(source: &mut impl Read) -> Option<MnistImages> {
        source.read_i32::<BigEndian>().ok().filter(|&x| x == 2051)?;
        let len = source
            .read_i32::<BigEndian>()
            .ok()
            .and_then(|x| usize::try_from(x).ok())?;
        let width = source
            .read_i32::<BigEndian>()
            .ok()
            .and_then(|x| usize::try_from(x).ok())?;
        let height = source
            .read_i32::<BigEndian>()
            .ok()
            .and_then(|x| usize::try_from(x).ok())?;
        let mut data = Vec::with_capacity(len * width * height);
        source.read_to_end(&mut data).ok()?;

        let images = Array::from_shape_vec((len, height, width), data).ok()?;

        Some(MnistImages {
            width,
            height,
            images,
        })
    }

    pub fn to_data(&self) -> Array2<f32> {
        self.images
            .to_shared()
            .reshape((self.images.len_of(Axis(0)), self.width * self.height))
            .mapv(|x| x as f32 / 255.)
    }
}
