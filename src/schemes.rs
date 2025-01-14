/**
This modules gives the tooling necessary to parse a sequence of tokens into a list of entities.
*/
use enum_iterator::Sequence;
use std::convert::TryFrom;
use std::error::Error;
use std::fmt::{Debug, Display};
use std::str::FromStr;

#[derive(Debug, PartialEq, Hash, Clone, Sequence, Eq)]
/// The inner prefix are the actual prefix that can be supplied by the user. All user prefix are of
/// length 1.
pub(crate) enum UserPrefix {
    I,
    O,
    B,
    E,
    S,
    U,
    L,
}

impl FromStr for UserPrefix {
    type Err = ParsingError<String>;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "I" => Ok(Self::I),
            "O" => Ok(Self::O),
            "B" => Ok(Self::B),
            "E" => Ok(Self::E),
            "S" => Ok(Self::S),
            "U" => Ok(Self::U),
            "L" => Ok(Self::L),
            _ => Err(ParsingError::PrefixError(String::from(s))),
        }
    }
}

impl TryFrom<char> for UserPrefix {
    type Error = ParsingError<String>;
    fn try_from(value: char) -> Result<Self, Self::Error> {
        match value {
            'I' => Ok(Self::I),
            'O' => Ok(Self::O),
            'B' => Ok(Self::B),
            'E' => Ok(Self::E),
            'S' => Ok(Self::S),
            'U' => Ok(Self::U),
            'L' => Ok(Self::L),
            _ => Err(ParsingError::PrefixError(String::from(value))),
        }
    }
}

impl From<UserPrefix> for Prefix {
    fn from(value: UserPrefix) -> Self {
        match value {
            UserPrefix::I => Self::I,
            UserPrefix::O => Self::O,
            UserPrefix::B => Self::B,
            UserPrefix::E => Self::E,
            UserPrefix::S => Self::S,
            UserPrefix::U => Self::U,
            UserPrefix::L => Self::L,
        }
    }
}

impl From<&UserPrefix> for Prefix {
    fn from(value: &UserPrefix) -> Self {
        match value {
            UserPrefix::I => Self::I,
            UserPrefix::O => Self::O,
            UserPrefix::B => Self::B,
            UserPrefix::E => Self::E,
            UserPrefix::S => Self::S,
            UserPrefix::U => Self::U,
            UserPrefix::L => Self::L,
        }
    }
}

#[derive(Debug, PartialEq, Hash, Clone, Sequence, Eq)]
/// Prefix represent an annotation specifying the place of a token in a chunk. For example, in
/// `IOB1`, the `I` prefix is used to indicate that the token is inside a NER. Prefix can only be a
/// single ascii character.
pub(crate) enum Prefix {
    I,
    O,
    B,
    E,
    S,
    U,
    L,
    /// The `ANY` prefix is more of a marker than a real prefix. It is not suppposed to be supplied
    /// by the user.
    Any,
}

impl Display for UserPrefix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Display for Prefix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Prefix {
    /// This functions verifies that this prefix and the other prefix are the same or one of them
    /// is the `PrefixAny` prefix.
    ///
    /// * `other`: The prefix to compare with `self`
    fn are_the_same_or_contains_any(&self, other: &Prefix) -> bool {
        match (self, other) {
            (&Prefix::Any, _) => true,
            (_, &Prefix::Any) => true,
            (s, o) if s == o => true,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Could not parse the string into a `Prefix`
pub enum ParsingError<S: AsRef<str>> {
    PrefixError(S),
    EmptyToken,
}

impl<S: AsRef<str>> Display for ParsingError<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PrefixError(s) => {
                let content = s.as_ref();
                write!(
                    f,
                    "Could not parse the following string into a Prefix: {}",
                    content
                )
            }
            Self::EmptyToken => {
                write!(f, "Received an empty string/&str")
            }
        }
    }
}
impl<S: AsRef<str> + Error> Error for ParsingError<S> {}

//TODO: Remove this impl and use UserPrefix instead
impl FromStr for Prefix {
    type Err = ParsingError<String>;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::try_from_with_static_error(s)
    }
}

//TODO: Remove this impl and use UserPrefix instead
impl Prefix {
    fn try_from_with_static_error(value: &str) -> Result<Self, ParsingError<String>> {
        match value {
            "I" => Ok(Prefix::I),
            "O" => Ok(Prefix::O),
            "B" => Ok(Prefix::B),
            "E" => Ok(Prefix::E),
            "S" => Ok(Prefix::S),
            "U" => Ok(Prefix::U),
            "L" => Ok(Prefix::L),
            _ => Err(ParsingError::PrefixError(String::from(value))),
        }
    }
}

#[derive(Debug, PartialEq, Hash, Clone)]
enum Tag {
    Same,
    Diff,
    Any,
}

#[derive(Debug, PartialEq, Hash, Clone)]
/// Inner structure used to define what a token is.
pub(super) struct InnerToken<'a> {
    /// The full token, such as `"B-PER"`, `"I-LOC"`, etc.
    pub(super) token: &'a str,
    /// The prefix, such as `B`, `I`, `O`, etc.
    pub(super) prefix: UserPrefix,
    /// The tag, such as '"PER"', '"LOC"'
    pub(super) tag: &'a str,
}

impl Default for InnerToken<'_> {
    fn default() -> Self {
        InnerToken {
            token: "",
            prefix: UserPrefix::I,
            tag: "",
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, PartialOrd, Ord)]
/// This enum represents the positon of the Prefix in a token. It assumes the length of the prefix
/// is 1.
enum PrefixCharIndex {
    /// This variant indicates that the prefix is located at the start of the token
    Start(usize),
    /// This variant indicates that the prefix is located at the end of the token and contains the
    /// number of chars in the token.
    End(usize, usize),
}
impl PrefixCharIndex {
    /// There are 3 cases we need to take care of:
    /// 1. Suffix is true and the format is "delimiter+prefix", where delimiter is a SINGLE
    ///    unicode char, such as '-'.
    /// 2. Suffix is false and the format is "prefix+delimiter", where the delimiter is a SINGLE
    ///    unicode char, such as '-'.
    /// 3. There is no tag, only prefixes. Therefore, every token is of length 1 and the suffix
    ///    argument is of no consequence.
    fn try_new(suffix: bool, token: &str) -> Result<Self, ParsingError<String>> {
        if suffix {
            let count = token.chars().count();
            if count == 0 {
                return Err(ParsingError::EmptyToken);
            }
            Ok(PrefixCharIndex::End(count - 1, count))
        } else {
            Ok(PrefixCharIndex::Start(0))
        }
    }
    fn to_index(&self) -> usize {
        match self {
            Self::Start(start) => *start,
            Self::End(end, _) => *end,
        }
    }
}

impl<'a> InnerToken<'a> {
    /// Create an InnerToken
    ///
    /// * `token`: str or String to parse the InnerToken from
    /// * `suffix`: Marker indicating if prefix is located at the end (when suffix is true) or the
    ///   end (when suffix is false) of the token
    /// * `delimiter`: Indicates the char used to separate the Prefix from the rest of the tag
    #[inline(always)]
    pub(super) fn try_new(token: &'a str, suffix: bool) -> Result<Self, ParsingError<String>> {
        let prefix_index_struct = PrefixCharIndex::try_new(suffix, token)?;
        let prefix_char_index = prefix_index_struct.to_index();
        let char_res = token.chars().nth(prefix_char_index);
        let prefix = match char_res {
            Some(c) => UserPrefix::try_from(c)?,
            None => return Err(ParsingError::PrefixError(String::from(token))),
        };
        if token.chars().count() == 1 {
            return Ok(Self {
                token,
                prefix,
                tag: "_",
            });
        };
        let mut tag = match prefix_index_struct {
            PrefixCharIndex::Start(_) => {
                let (offset, _) = token.char_indices().nth(2).unwrap();
                &token[offset..]
            }
            PrefixCharIndex::End(_, _count @ 1) => token,
            PrefixCharIndex::End(_, _count @ 2) => {
                return Err(ParsingError::PrefixError(String::from(token)))
            }
            PrefixCharIndex::End(_, count) => {
                let (_, (offset, _)) = token
                    .char_indices()
                    .enumerate()
                    .take_while(|(i, _)| i < &(count - 1))
                    .last()
                    .unwrap();
                &token[0..offset]
            }
        };
        if tag.is_empty() {
            tag = "_";
        }
        Ok(Self { token, prefix, tag })
    }

    #[inline]
    fn check_tag(&self, prev: &InnerToken, cond: &Tag) -> bool {
        match cond {
            Tag::Any => true,
            Tag::Same if prev.tag == self.tag => true,
            Tag::Diff if prev.tag != self.tag => true,
            _ => false,
        }
    }
    /// Check whether the prefix patterns are matched.
    ///
    /// * `prev`: Previous token
    /// * `patterns`: Patterns to match the token against
    fn check_patterns(
        &self,
        prev: &InnerToken,
        patterns_to_check: &[(Prefix, Prefix, Tag)],
    ) -> bool {
        for (prev_prefix, current_prefix, tag_cond) in patterns_to_check {
            if prev_prefix.are_the_same_or_contains_any(&Prefix::from(prev.prefix.clone()))
                && current_prefix.are_the_same_or_contains_any(&Prefix::from(self.prefix.clone()))
                && self.check_tag(prev, tag_cond)
            {
                return true;
            }
        }
        false
    }
}

#[derive(Debug, Clone, Copy, Sequence, Hash, Eq, PartialEq)]
/// Enumeration of the supported Schemes. They are used to indicate how we are
/// supposed to parse and chunk the different tokens.
#[derive(Default)]
pub enum SchemeType {
    IOB1,
    IOE1,
    #[default]
    IOB2,
    IOE2,
    IOBES,
    BILOU,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Encountered an invalid token when parsing the entities.
pub struct InvalidToken(pub String);

impl Display for InvalidToken {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Invalid token: {}", self.0)
    }
}

impl Error for InvalidToken {}

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, PartialEq)]
pub(super) enum Token<'a> {
    IOB1 { token: InnerToken<'a> },
    IOE1 { token: InnerToken<'a> },
    IOB2 { token: InnerToken<'a> },
    IOE2 { token: InnerToken<'a> },
    IOBES { token: InnerToken<'a> },
    BILOU { token: InnerToken<'a> },
}

impl Default for Token<'_> {
    fn default() -> Self {
        Token::IOB1 {
            token: InnerToken::default(),
        }
    }
}

impl<'a> Token<'a> {
    pub(crate) const IOB1_ALLOWED_PREFIXES: [Prefix; 3] = [Prefix::I, Prefix::O, Prefix::B];
    const IOB1_START_PATTERNS: [(Prefix, Prefix, Tag); 5] = [
        (Prefix::O, Prefix::I, Tag::Any),
        (Prefix::I, Prefix::I, Tag::Diff),
        (Prefix::B, Prefix::I, Tag::Any),
        (Prefix::I, Prefix::B, Tag::Same),
        (Prefix::B, Prefix::B, Tag::Same),
    ];
    const IOB1_INSIDE_PATTERNS: [(Prefix, Prefix, Tag); 2] = [
        (Prefix::B, Prefix::I, Tag::Same),
        (Prefix::I, Prefix::I, Tag::Same),
    ];
    const IOB1_END_PATTERNS: [(Prefix, Prefix, Tag); 6] = [
        (Prefix::I, Prefix::I, Tag::Diff),
        (Prefix::I, Prefix::O, Tag::Any),
        (Prefix::I, Prefix::B, Tag::Any),
        (Prefix::B, Prefix::O, Tag::Any),
        (Prefix::B, Prefix::I, Tag::Diff),
        (Prefix::B, Prefix::B, Tag::Same),
    ];
    pub(crate) const IOE1_ALLOWED_PREFIXES: [Prefix; 3] = [Prefix::I, Prefix::O, Prefix::E];
    const IOE1_START_PATTERNS: [(Prefix, Prefix, Tag); 4] = [
        (Prefix::O, Prefix::I, Tag::Any),
        (Prefix::I, Prefix::I, Tag::Diff),
        (Prefix::E, Prefix::I, Tag::Any),
        (Prefix::E, Prefix::E, Tag::Same),
    ];
    const IOE1_INSIDE_PATTERNS: [(Prefix, Prefix, Tag); 2] = [
        (Prefix::I, Prefix::I, Tag::Same),
        (Prefix::I, Prefix::E, Tag::Same),
    ];
    const IOE1_END_PATTERNS: [(Prefix, Prefix, Tag); 5] = [
        (Prefix::I, Prefix::I, Tag::Diff),
        (Prefix::I, Prefix::O, Tag::Any),
        (Prefix::I, Prefix::E, Tag::Diff),
        (Prefix::E, Prefix::I, Tag::Same),
        (Prefix::E, Prefix::E, Tag::Same),
    ];

    pub(crate) const IOB2_ALLOWED_PREFIXES: [Prefix; 3] = [Prefix::I, Prefix::O, Prefix::B];
    const IOB2_START_PATTERNS: [(Prefix, Prefix, Tag); 1] = [(Prefix::Any, Prefix::B, Tag::Any)];
    const IOB2_INSIDE_PATTERNS: [(Prefix, Prefix, Tag); 2] = [
        (Prefix::B, Prefix::I, Tag::Same),
        (Prefix::I, Prefix::I, Tag::Same),
    ];
    const IOB2_END_PATTERNS: [(Prefix, Prefix, Tag); 6] = [
        (Prefix::I, Prefix::O, Tag::Any),
        (Prefix::I, Prefix::I, Tag::Diff),
        (Prefix::I, Prefix::B, Tag::Any),
        (Prefix::B, Prefix::O, Tag::Any),
        (Prefix::B, Prefix::I, Tag::Diff),
        (Prefix::B, Prefix::B, Tag::Any),
    ];
    pub(crate) const IOE2_ALLOWED_PREFIXES: [Prefix; 3] = [Prefix::I, Prefix::O, Prefix::E];
    const IOE2_START_PATTERNS: [(Prefix, Prefix, Tag); 6] = [
        (Prefix::O, Prefix::I, Tag::Any),
        (Prefix::O, Prefix::E, Tag::Any),
        (Prefix::E, Prefix::I, Tag::Any),
        (Prefix::E, Prefix::E, Tag::Any),
        (Prefix::I, Prefix::I, Tag::Diff),
        (Prefix::I, Prefix::E, Tag::Diff),
    ];
    const IOE2_INSIDE_PATTERNS: [(Prefix, Prefix, Tag); 2] = [
        (Prefix::I, Prefix::E, Tag::Same),
        (Prefix::I, Prefix::I, Tag::Same),
    ];
    const IOE2_END_PATTERNS: [(Prefix, Prefix, Tag); 1] = [(Prefix::E, Prefix::Any, Tag::Any)];

    pub(crate) const IOBES_ALLOWED_PREFIXES: [Prefix; 5] =
        [Prefix::I, Prefix::O, Prefix::E, Prefix::B, Prefix::S];
    const IOBES_START_PATTERNS: [(Prefix, Prefix, Tag); 4] = [
        (Prefix::B, Prefix::I, Tag::Same),
        (Prefix::B, Prefix::E, Tag::Same),
        (Prefix::I, Prefix::I, Tag::Same),
        (Prefix::I, Prefix::E, Tag::Same),
    ];
    const IOBES_INSIDE_PATTERNS: [(Prefix, Prefix, Tag); 2] = [
        (Prefix::S, Prefix::Any, Tag::Any),
        (Prefix::E, Prefix::Any, Tag::Any),
    ];
    const IOBES_END_PATTERNS: [(Prefix, Prefix, Tag); 2] = [
        (Prefix::S, Prefix::Any, Tag::Any),
        (Prefix::E, Prefix::Any, Tag::Any),
    ];

    pub(crate) const BILOU_ALLOWED_PREFIXES: [Prefix; 5] =
        [Prefix::I, Prefix::O, Prefix::U, Prefix::B, Prefix::O];
    const BILOU_START_PATTERNS: [(Prefix, Prefix, Tag); 2] = [
        (Prefix::Any, Prefix::B, Tag::Any),
        (Prefix::Any, Prefix::U, Tag::Any),
    ];
    const BILOU_INSIDE_PATTERNS: [(Prefix, Prefix, Tag); 4] = [
        (Prefix::B, Prefix::I, Tag::Same),
        (Prefix::B, Prefix::L, Tag::Same),
        (Prefix::I, Prefix::I, Tag::Same),
        (Prefix::I, Prefix::L, Tag::Same),
    ];
    const BILOU_END_PATTERNS: [(Prefix, Prefix, Tag); 2] = [
        (Prefix::U, Prefix::Any, Tag::Any),
        (Prefix::L, Prefix::Any, Tag::Any),
    ];

    pub(crate) fn new(scheme: SchemeType, token: InnerToken<'a>) -> Self {
        match scheme {
            SchemeType::IOB1 => Token::IOB1 { token },
            SchemeType::IOB2 => Token::IOB2 { token },
            SchemeType::IOE1 => Token::IOE1 { token },
            SchemeType::IOE2 => Token::IOE2 { token },
            SchemeType::IOBES => Token::IOBES { token },
            SchemeType::BILOU => Token::BILOU { token },
        }
    }
    fn allowed_prefixes(&'a self) -> &'static [Prefix] {
        match self {
            Self::IOB1 { .. } => &Self::IOB1_ALLOWED_PREFIXES,
            Self::IOE1 { .. } => &Self::IOE1_ALLOWED_PREFIXES,
            Self::IOB2 { .. } => &Self::IOB2_ALLOWED_PREFIXES,
            Self::IOE2 { .. } => &Self::IOE2_ALLOWED_PREFIXES,
            Self::IOBES { .. } => &Self::IOBES_ALLOWED_PREFIXES,
            Self::BILOU { .. } => &Self::BILOU_ALLOWED_PREFIXES,
        }
    }
    fn start_patterns(&'a self) -> &'static [(Prefix, Prefix, Tag)] {
        match self {
            Self::IOB1 { .. } => &Self::IOB1_START_PATTERNS,
            Self::IOE1 { .. } => &Self::IOE1_START_PATTERNS,
            Self::IOB2 { .. } => &Self::IOB2_START_PATTERNS,
            Self::IOE2 { .. } => &Self::IOE2_START_PATTERNS,
            Self::IOBES { .. } => &Self::IOBES_START_PATTERNS,
            Self::BILOU { .. } => &Self::BILOU_START_PATTERNS,
        }
    }
    fn inside_patterns(&'a self) -> &'static [(Prefix, Prefix, Tag)] {
        match self {
            Self::IOB1 { .. } => &Self::IOB1_INSIDE_PATTERNS,
            Self::IOE1 { .. } => &Self::IOE1_INSIDE_PATTERNS,
            Self::IOB2 { .. } => &Self::IOB2_INSIDE_PATTERNS,
            Self::IOE2 { .. } => &Self::IOE2_INSIDE_PATTERNS,
            Self::IOBES { .. } => &Self::IOBES_INSIDE_PATTERNS,
            Self::BILOU { .. } => &Self::BILOU_INSIDE_PATTERNS,
        }
    }
    fn end_patterns(&'a self) -> &'static [(Prefix, Prefix, Tag)] {
        match self {
            Self::IOB1 { .. } => &Self::IOB1_END_PATTERNS,
            Self::IOE1 { .. } => &Self::IOE1_END_PATTERNS,
            Self::IOB2 { .. } => &Self::IOB2_END_PATTERNS,
            Self::IOE2 { .. } => &Self::IOE2_END_PATTERNS,
            Self::IOBES { .. } => &Self::IOBES_END_PATTERNS,
            Self::BILOU { .. } => &Self::BILOU_END_PATTERNS,
        }
    }

    pub(super) fn inner(&self) -> &InnerToken {
        match self {
            Self::IOE1 { token } => token,
            Self::IOE2 { token } => token,
            Self::IOB1 { token } => token,
            Self::IOB2 { token } => token,
            Self::BILOU { token } => token,
            Self::IOBES { token } => token,
        }
    }

    pub(super) fn is_valid(&self) -> bool {
        self.allowed_prefixes()
            .contains(&Prefix::from(self.inner().prefix.clone()))
    }

    /// Check whether the current token is the start of chunk.
    pub(super) fn is_start(&self, prev: &InnerToken) -> bool {
        match self {
            Self::IOB1 { token } => token.check_patterns(prev, self.start_patterns()),
            Self::IOB2 { token } => token.check_patterns(prev, self.start_patterns()),
            Self::IOE1 { token } => token.check_patterns(prev, self.start_patterns()),
            Self::IOE2 { token } => token.check_patterns(prev, self.start_patterns()),
            Self::IOBES { token } => token.check_patterns(prev, self.start_patterns()),
            Self::BILOU { token } => token.check_patterns(prev, self.start_patterns()),
        }
    }
    /// Check whether the current token is the inside of chunk.
    pub(super) fn is_inside(&self, prev: &InnerToken) -> bool {
        match self {
            Self::IOB1 { token } => token.check_patterns(prev, self.inside_patterns()),
            Self::IOB2 { token } => token.check_patterns(prev, self.inside_patterns()),
            Self::IOE1 { token } => token.check_patterns(prev, self.inside_patterns()),
            Self::IOE2 { token } => token.check_patterns(prev, self.inside_patterns()),
            Self::IOBES { token } => token.check_patterns(prev, self.inside_patterns()),
            Self::BILOU { token } => token.check_patterns(prev, self.inside_patterns()),
        }
    }
    /// Check whether the *previous* token is the end of chunk.
    pub(super) fn is_end(&self, prev: &InnerToken) -> bool {
        match self {
            Self::IOB1 { token } => token.check_patterns(prev, self.end_patterns()),
            Self::IOB2 { token } => token.check_patterns(prev, self.end_patterns()),
            Self::IOE1 { token } => token.check_patterns(prev, self.end_patterns()),
            Self::IOE2 { token } => token.check_patterns(prev, self.end_patterns()),
            Self::IOBES { token } => token.check_patterns(prev, self.end_patterns()),
            Self::BILOU { token } => token.check_patterns(prev, self.end_patterns()),
        }
    }
    pub(super) fn get_tag(&mut self) -> &'a str {
        match self {
            Self::IOB1 { token } => token.tag,
            Self::IOE1 { token } => token.tag,
            Self::IOB2 { token } => token.tag,
            Self::IOE2 { token } => token.tag,
            Self::IOBES { token } => token.tag,
            Self::BILOU { token } => token.tag,
        }
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use enum_iterator::all;
    use quickcheck::{self, TestResult};

    impl quickcheck::Arbitrary for Prefix {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            let mut choice_slice: Vec<Prefix> = all::<Prefix>().collect();
            // Removes the `ALL` prefix
            choice_slice.swap_remove(choice_slice.len() - 1);
            g.choose(choice_slice.as_ref()).unwrap().clone()
        }
    }

    impl quickcheck::Arbitrary for UserPrefix {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            let mut choice_slice: Vec<UserPrefix> = all::<UserPrefix>().collect();
            // Removes the `ALL` prefix
            choice_slice.swap_remove(choice_slice.len() - 1);
            g.choose(choice_slice.as_ref()).unwrap().clone()
        }
    }

    #[test]
    fn test_propertie_test_new_token_len() {
        fn propertie_test_new_token_len(
            chars: Vec<char>,
            prefix: UserPrefix,
            suffix: bool,
            delimiter: char,
        ) -> TestResult {
            let tag: String = chars.into_iter().collect();
            if tag.len() == 0 {
                return TestResult::discard();
            };
            let tag_len = tag.len();
            dbg!(tag_len);
            let token_string = if suffix {
                tag + &String::from(delimiter) + &prefix.to_string()
            } else {
                prefix.to_string() + &String::from(delimiter) + &tag
            };
            let token: &str = token_string.as_ref();
            dbg!(token);
            let inner_token_res = InnerToken::try_new(token, suffix);
            match inner_token_res {
                Err(err) => match err {
                    ParsingError::PrefixError(s) => panic!("{}", ParsingError::PrefixError(s)),
                    ParsingError::EmptyToken => TestResult::discard(),
                },
                Ok(inner_token) => {
                    if inner_token.tag.len() == tag_len {
                        TestResult::passed()
                    } else {
                        TestResult::failed()
                    }
                }
            }
        }
        let mut qc = quickcheck::QuickCheck::new().tests(2000);
        qc.quickcheck(
            propertie_test_new_token_len as fn(Vec<char>, UserPrefix, bool, char) -> TestResult,
        );
    }

    #[test]
    fn test_propertie_test_new_token_prefix() {
        fn propertie_test_new_token_prefix(
            chars: Vec<char>,
            prefix: UserPrefix,
            suffix: bool,
            delimiter: char,
        ) -> TestResult {
            let tag: String = chars.into_iter().collect();
            if tag.len() == 0 {
                return TestResult::discard();
            };
            let token_string = if suffix {
                tag + &String::from(delimiter) + &prefix.to_string()
            } else {
                prefix.to_string() + &String::from(delimiter) + &tag
            };
            let token: &str = token_string.as_ref();
            dbg!(token);
            let inner_token_res = InnerToken::try_new(token, suffix);
            match inner_token_res {
                Err(err) => match err {
                    ParsingError::PrefixError(s) => panic!("{}", ParsingError::PrefixError(s)),
                    ParsingError::EmptyToken => TestResult::discard(),
                },
                Ok(inner_token) => {
                    if inner_token.prefix == prefix {
                        TestResult::passed()
                    } else {
                        TestResult::failed()
                    }
                }
            }
        }

        let mut qc = quickcheck::QuickCheck::new().tests(2000);
        qc.quickcheck(
            propertie_test_new_token_prefix as fn(Vec<char>, UserPrefix, bool, char) -> TestResult,
        );
    }

    #[test]
    fn test_empty_token_return_err() {
        let token = "";
        let suffix = true;
        let res = InnerToken::try_new(token, suffix).is_err();
        assert!(res);
    }

    #[test]
    fn test_innertoken_new_with_suffix() {
        let tokens = vec![
            ("PER-I", "PER", UserPrefix::I),
            ("PER-B", "PER", UserPrefix::B),
            ("LOC-I", "LOC", UserPrefix::I),
            ("O", "_", UserPrefix::O),
        ];
        let suffix = true;
        for (i, (token, tag, prefix)) in tokens.into_iter().enumerate() {
            let inner_token = InnerToken::try_new(token, suffix).unwrap();
            let expected_inner_token = InnerToken { token, prefix, tag };
            dbg!(i);
            assert_eq!(inner_token, expected_inner_token)
        }
    }
    #[test]
    fn test_innertoken_new() {
        let tokens = vec![
            ("I-PER", "PER", UserPrefix::I),
            ("B-PER", "PER", UserPrefix::B),
            ("I-LOC", "LOC", UserPrefix::I),
            ("O", "_", UserPrefix::O),
        ];
        let suffix = false;
        for (i, (token, tag, prefix)) in tokens.into_iter().enumerate() {
            let inner_token = InnerToken::try_new(token, suffix).unwrap();
            let expected_inner_token = InnerToken { token, prefix, tag };
            dbg!(i);
            assert_eq!(inner_token, expected_inner_token)
        }
    }

    #[test]
    /// This function tests an implicit invariant used when building `Tokens` and `UnicodeIndex`:
    /// The prefix must be composed of a single unicode character.
    fn test_prefix_length() {
        let all_prefixes: Vec<_> = all::<UserPrefix>().collect();
        for p in all_prefixes {
            let len = p.to_string().as_str().len();
            assert_eq!(1, len);
        }
    }
}
