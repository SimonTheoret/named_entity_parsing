use crate::{schemes::SchemeType, InnerToken, UserPrefix};
use ahash::AHashSet;
use std::sync::LazyLock;

static ALLOWED_IOB2_PREFIXES: LazyLock<[AHashSet<UserPrefix>; 4]> = LazyLock::new(|| {
    [
        AHashSet::from([UserPrefix::I, UserPrefix::O, UserPrefix::B]),
        AHashSet::from([UserPrefix::I, UserPrefix::B]),
        AHashSet::from([UserPrefix::B, UserPrefix::O]),
        AHashSet::from([UserPrefix::B]),
    ]
});

static ALLOWED_IOE2_PREFIXES: LazyLock<[AHashSet<UserPrefix>; 4]> = LazyLock::new(|| {
    [
        AHashSet::from([UserPrefix::I, UserPrefix::O, UserPrefix::E]),
        AHashSet::from([UserPrefix::I, UserPrefix::E]),
        AHashSet::from([UserPrefix::E, UserPrefix::O]),
        AHashSet::from([UserPrefix::E]),
    ]
});
static ALLOWED_IOBES_PREFIXES: LazyLock<[AHashSet<UserPrefix>; 9]> = LazyLock::new(|| {
    [
        AHashSet::from([
            UserPrefix::I,
            UserPrefix::O,
            UserPrefix::B,
            UserPrefix::E,
            UserPrefix::S,
        ]),
        AHashSet::from([UserPrefix::I, UserPrefix::B, UserPrefix::E, UserPrefix::S]),
        AHashSet::from([UserPrefix::I, UserPrefix::O, UserPrefix::B, UserPrefix::E]),
        AHashSet::from([UserPrefix::O, UserPrefix::B, UserPrefix::E, UserPrefix::S]),
        AHashSet::from([UserPrefix::I, UserPrefix::B, UserPrefix::E]),
        AHashSet::from([UserPrefix::B, UserPrefix::E, UserPrefix::S]),
        AHashSet::from([UserPrefix::O, UserPrefix::B, UserPrefix::E]),
        AHashSet::from([UserPrefix::B, UserPrefix::E]),
        AHashSet::from([UserPrefix::S]),
    ]
});

static ALLOWED_BILOU_PREFIXES: LazyLock<[AHashSet<UserPrefix>; 9]> = LazyLock::new(|| {
    [
        AHashSet::from([
            UserPrefix::I,
            UserPrefix::O,
            UserPrefix::B,
            UserPrefix::L,
            UserPrefix::U,
        ]),
        AHashSet::from([UserPrefix::I, UserPrefix::B, UserPrefix::L, UserPrefix::U]),
        AHashSet::from([UserPrefix::I, UserPrefix::O, UserPrefix::B, UserPrefix::L]),
        AHashSet::from([UserPrefix::O, UserPrefix::B, UserPrefix::L, UserPrefix::U]),
        AHashSet::from([UserPrefix::I, UserPrefix::B, UserPrefix::L]),
        AHashSet::from([UserPrefix::B, UserPrefix::L, UserPrefix::U]),
        AHashSet::from([UserPrefix::O, UserPrefix::B, UserPrefix::L]),
        AHashSet::from([UserPrefix::B, UserPrefix::L]),
        AHashSet::from([UserPrefix::U]),
    ]
});

/// This impl block contains the logic of the auto-detect feature.
impl SchemeType {
    /// Autodetect the scheme used in `sequences`. Note that this function has to parse the input,
    /// so the longer the input, the longer this function will take. It might be better to give it
    /// a relatively small sample and to call only once. This function cannot return a false
    /// positive.
    ///
    /// `try_auto_detect` supports the following
    /// schemes:
    /// - IOB2
    /// - IOE2
    /// - IOBES
    /// - BILOU
    pub fn try_auto_detect(sequences: &[Vec<&str>], suffix: bool) -> Option<SchemeType> {
        let mut prefixes: AHashSet<UserPrefix> = AHashSet::default();
        for tokens in sequences {
            for token in tokens {
                let tok = InnerToken::try_new(token, suffix);
                match tok {
                    Ok(p) => prefixes.insert(p.prefix),
                    Err(_) => continue,
                };
            }
        }
        if ALLOWED_IOB2_PREFIXES.contains(&prefixes) {
            return Some(SchemeType::IOB2);
        } else if ALLOWED_IOE2_PREFIXES.contains(&prefixes) {
            return Some(SchemeType::IOE2);
        } else if ALLOWED_BILOU_PREFIXES.contains(&prefixes) {
            return Some(SchemeType::BILOU);
        } else if ALLOWED_IOBES_PREFIXES.contains(&prefixes) {
            return Some(SchemeType::IOBES);
        };
        None
    }
}
#[cfg(test)]
mod test {
    use crate::SchemeType;

    #[test]
    fn test_auto_detect_scheme_by_prefix() {
        let inputs = vec![build_str_vec_diff(), build_str_vec()];
        let actual = SchemeType::try_auto_detect(&inputs, false).unwrap();
        let expected = SchemeType::IOB2;
        assert_eq!(actual, expected)
    }

    fn build_str_vec() -> Vec<&'static str> {
        vec!["B-PER", "I-PER", "O", "B-LOC"]
    }
    fn build_str_vec_diff() -> Vec<&'static str> {
        vec![
            "B-GEO", "I-GEO", "O", "B-GEO", "O", "B-PER", "I-PER", "I-PER", "B-LOC",
        ]
    }
}
