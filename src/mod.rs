use crate::entity::schemes::{InnerToken, Token, UserPrefix};
use ahash::AHashSet;
use flatarray::FlatArray;
use std::{
    cell::{RefCell, UnsafeCell},
    cmp::Ordering,
    error::Error,
    fmt::{Debug, Display},
    mem::take,
    ops::{Deref, DerefMut},
    slice::Iter,
};

mod autodetect;
mod schemes;

// Re-exporting
pub use schemes::{InvalidToken, ParsingError, SchemeType};

/// An entity represent a named objet in named entity recognition (NER). It contains a start and an
/// end(i.e. at what index of the list does it starts and ends) and a tag, which the associated
/// entity (such as `LOC`, `NAME`, `PER`, etc.). It is important to note that the `end` field
/// differ from the value used in SeqEval when `strict = false`.
#[derive(Debug, Hash, Clone, PartialEq, Eq, PartialOrd)]
pub struct Entity<'a> {
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) tag: &'a str,
}

impl<'a> Entity<'a> {
    pub(crate) fn new(start: usize, end: usize, tag: &'a str) -> Self {
        Entity { start, end, tag }
    }
}

impl Display for Entity<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}, {})", self.tag, self.start, self.end)
    }
}

/// Leniently retrieves the entities from a sequence.
#[inline(always)]
pub(crate) fn get_entities_lenient<'a>(
    sequence: &'a FlatArray<&'a str>,
    suffix: bool,
) -> Result<Entities<'a>, ParsingError<String>> {
    let mut res = Vec::with_capacity(sequence.len() / 2);
    let mut indices = Vec::with_capacity(sequence.len() / 2);
    indices.push(0);
    for vec_of_chunks in sequence.iter_arrays() {
        let chunk_iter = LenientChunkIter::new(vec_of_chunks, suffix);
        indices.push(vec_of_chunks.len());
        for entity in chunk_iter {
            res.push(entity?);
        }
    }
    Ok(Entities(FlatArray::from_raw(res, indices)))
    o
}

/// This wrapper around the content iterator appends a single `"O"` at the end of its inner
/// iterator.
struct InnerLenientChunkIter<'a> {
    content: Iter<'a, &'a str>,
    is_at_end: bool,
}

impl<'a> InnerLenientChunkIter<'a> {
    fn new(seq: &'a [&'a str]) -> Self {
        InnerLenientChunkIter {
            content: seq.iter(),
            is_at_end: false,
        }
    }
}

impl<'a> Iterator for InnerLenientChunkIter<'a> {
    type Item = &'a str;
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let next_value = self.content.next();
        if next_value.is_none() {
            match self.is_at_end {
                // End of iteration
                true => None,
                // Reached end of iterator. Need to add a `"O"` at the end
                false => {
                    self.is_at_end = true;
                    Some("O")
                }
            }
        } else {
            next_value.copied()
        }
    }
}

/// This struct iterates over a *single* sequence and returns the chunks associated with it.
struct LenientChunkIter<'a> {
    /// The content on which we are iterating
    inner: InnerLenientChunkIter<'a>,
    /// The prefix of the previous chunk (e.g. 'I')
    prev_prefix: UserPrefix,
    /// The type of the previous chunk (e.g. `"PER"`)
    prev_type: Option<&'a str>,
    begin_offset: usize,
    suffix: bool,
    index: usize,
}

impl<'a> LenientChunkIter<'a> {
    fn new(sequence: &'a [&'a str], suffix: bool) -> Self {
        LenientChunkIter {
            inner: InnerLenientChunkIter::new(sequence),
            prev_type: None,
            prev_prefix: UserPrefix::O,
            begin_offset: 0,
            suffix,
            index: 0,
        }
    }
}

impl<'a> Iterator for LenientChunkIter<'a> {
    type Item = Result<Entity<'a>, ParsingError<String>>;
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let current_chunk = self.inner.next()?; // no more chunks. We are done
            let mut inner_token = match InnerToken::try_new(current_chunk, self.suffix) {
                Ok(v) => v,
                Err(e) => {
                    self.index += 1;
                    return Some(Err(e));
                }
            };
            let ret: Option<Self::Item>;
            if self.end_of_chunk(&inner_token.prefix, inner_token.tag) {
                ret = Some(Ok(Entity::new(
                    self.begin_offset,
                    self.index,
                    self.prev_type.unwrap(),
                )));
                self.prev_prefix = inner_token.prefix;
                self.prev_type = Some(inner_token.tag);
                self.index += 1;
                return ret;
            } else if self.start_of_chunk(&inner_token.prefix, inner_token.tag) {
                self.begin_offset = self.index;
            };
            self.prev_prefix = inner_token.prefix;
            self.prev_type = Some(take(&mut inner_token.tag));
            self.index += 1;
        }
    }
}
impl<'a> LenientChunkIter<'a> {
    //     tag -> prefix
    //     type -> classe
    ///     """Checks if a chunk ended between the previous and current word.
    #[inline(always)]
    fn end_of_chunk(&self, current_prefix: &UserPrefix, current_type: &'a str) -> bool {
        let wrapped_type = Some(current_type);
        // Cloning a prefix is very inexpensive
        match (self.prev_prefix.clone(), current_prefix) {
            (UserPrefix::E, _) => true,
            (UserPrefix::S, _) => true,
            (UserPrefix::B, UserPrefix::B) => true,
            (UserPrefix::B, UserPrefix::S) => true,
            (UserPrefix::B, UserPrefix::O) => true,
            (UserPrefix::I, UserPrefix::B) => true,
            (UserPrefix::I, UserPrefix::S) => true,
            (UserPrefix::I, UserPrefix::O) => true,
            (self_prefix, _) => {
                !matches!(self_prefix, UserPrefix::O) && self.prev_type != wrapped_type
            }
        }
    }

    /// Checks if a chunk started between the previous and current word.
    #[inline(always)]
    fn start_of_chunk(&self, current_prefix: &UserPrefix, current_type: &'a str) -> bool {
        let wrapped_type = Some(current_type);
        match (self.prev_prefix.clone(), current_prefix) {
            // Cloning a prefix is very inexpensive
            (_, UserPrefix::B) => true,
            (_, UserPrefix::S) => true,
            (UserPrefix::E, UserPrefix::E) => true,
            (UserPrefix::E, UserPrefix::I) => true,
            (UserPrefix::S, UserPrefix::E) => true,
            (UserPrefix::S, UserPrefix::I) => true,
            (UserPrefix::O, UserPrefix::E) => true,
            (UserPrefix::O, UserPrefix::I) => true,
            (_, curr_prefix) => {
                !matches!(curr_prefix, UserPrefix::O) && self.prev_type != wrapped_type
            }
        }
    }
}

/// This struct is capable of building efficiently the Tokens with a given outside_token. This
/// iterator avoids reallocation and keeps good ergonomic inside the `new` function of `Tokens`.
/// The `outside_token` field is the *last* token generated by this struct when calling `.next()`.
/// This struct is used to parse the tokens into an easier to use structs called `Token`s. During
/// iteration, it returns as last token the `'O'` tag.
struct ExtendedTokensIterator<'a> {
    outside_token: Token<'a>,
    tokens: &'a mut [&'a str],
    scheme: SchemeType,
    suffix: bool,
    index: usize,
    /// Total length to iterate over. *This length is equal to token.len()*
    total_len: usize,
}
impl<'a> Iterator for ExtendedTokensIterator<'a> {
    type Item = Result<Token<'a>, ParsingError<String>>;
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let ret = match self.index.cmp(&self.total_len) {
            Ordering::Greater => None,
            Ordering::Equal => Some(Ok(take(&mut self.outside_token))),
            Ordering::Less => {
                let str = unsafe { take(self.tokens.get_unchecked_mut(self.index)) };
                let inner_token = InnerToken::try_new(str, self.suffix);
                match inner_token {
                    Err(msg) => Some(Err(msg)),
                    Ok(res) => Some(Ok(Token::new(self.scheme, res))),
                }
            }
        };
        self.index += 1;
        ret
    }
}
impl<'a> ExtendedTokensIterator<'a> {
    fn new(
        outside_token: Token<'a>,
        tokens: &'a mut [&'a str],
        scheme: SchemeType,
        suffix: bool,
    ) -> Self {
        let total_len = tokens.len();
        Self {
            outside_token,
            tokens,
            scheme,
            suffix,
            index: 0,
            total_len,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Intermediary type used to build the entities of a Vec of cows. It is a wrapper around a Vec of
/// `Token` and allows us to parse itself.
struct Tokens<'a> {
    /// Extended tokens are the parsed list of token with an `O` token as first token.
    extended_tokens: Vec<Token<'a>>,
}
impl<'a> Tokens<'a> {
    #[inline(always)]
    fn new(
        tokens: &'a mut [&'a str],
        scheme: SchemeType,
        suffix: bool,
    ) -> Result<Self, ParsingError<String>> {
        let outside_token_inner = InnerToken::try_new("O", suffix)?;
        let outside_token = Token::new(scheme, outside_token_inner);
        let tokens_iter = ExtendedTokensIterator::new(outside_token, tokens, scheme, suffix);
        let extended_tokens: Result<Vec<Token>, ParsingError<String>> = tokens_iter.collect();
        match extended_tokens {
            Err(prefix_error) => Err(prefix_error),
            Ok(tokens) => Ok(Self {
                extended_tokens: tokens,
            }),
        }
    }

    /// Returns the index + 1 of the last token inside the current chunk when given a `start` index and
    /// the previous token. It allows us to call `next = Tokens[start, self.forward(i, prev)]`>
    ///
    /// * `start`: Indexing at which we are starting to look for a token not inside.
    /// * `prev`: Previous token. This token is necessary to know if the token at index `start` is
    ///    inside or not.
    #[inline(always)]
    fn forward(&self, start: usize, prev: &Token<'a>) -> usize {
        let slice_of_interest = &self.extended_tokens()[start..];
        let mut swap_token = prev;
        for (i, current_token) in slice_of_interest.iter().enumerate() {
            if current_token.is_inside(swap_token.inner()) {
                swap_token = current_token;
            } else {
                return i + start;
            }
        }
        &self.extended_tokens.len() - 2
    }

    /// This method returns a bool if the token at index `i` is *NOT*
    /// part of the same chunk as token at `i-1` or is not part of a
    /// chunk at all. Else, it returns false
    ///
    /// * `i`: Index of the token.
    #[inline(always)]
    fn is_end(&self, i: usize) -> bool {
        let token = &self.extended_tokens()[i];
        let prev = &self.extended_tokens()[i - 1];
        token.is_end(prev.inner())
    }

    #[inline(always)]
    fn extended_tokens(&'a self) -> &'a Vec<Token<'a>> {
        let res: &Vec<Token> = self.extended_tokens.as_ref();
        res
    }
}

/// Iterator and adaptor for iterating over the `Entities` of a Tokens struct.
///
/// * `index`: Index of the current iteration
/// * `current`: Current token
/// * `prev`:  Previous token
/// * `prev_prev`: Previous token of the previous token
struct EntitiesIterAdaptor<'a> {
    index: usize,
    tokens: RefCell<Tokens<'a>>,
    len: usize,
}

impl<'a> Iterator for EntitiesIterAdaptor<'a> {
    type Item = Option<Result<Entity<'a>, InvalidToken>>;
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let ret: Option<Option<Result<Entity<'a>, InvalidToken>>>;
        if self.index >= self.len - 1 {
            return None;
        }
        let mut_tokens = &self.tokens;
        let mut mut_tokens_ref = mut_tokens.borrow_mut();
        let (current_pre_ref_cell, prev) =
            unsafe { Self::take_out_pair(&mut mut_tokens_ref, self.index) };
        let current = RefCell::new(current_pre_ref_cell);
        let borrowed_current = current.borrow();
        let is_valid = borrowed_current.is_valid();
        if !is_valid {
            ret = Some(Some(Err(InvalidToken(
                borrowed_current.inner().token.to_string(),
            ))))
        } else if borrowed_current.is_start(prev.inner()) {
            drop(mut_tokens_ref);
            let end = mut_tokens
                .borrow()
                .forward(self.index + 1, &borrowed_current);
            if mut_tokens.borrow().is_end(end) {
                drop(borrowed_current);
                let tag = current.into_inner().get_tag();
                let entity = Entity {
                    start: self.index,
                    end,
                    tag,
                };
                self.index = end;
                ret = Some(Some(Ok(entity)));
            } else {
                self.index = end;
                ret = Some(None);
            }
        } else {
            self.index += 1;
            ret = Some(None);
        };
        ret
    }
}
impl<'a, 'b> EntitiesIterAdaptor<'a>
where
    'a: 'b,
{
    /// Takes out the current tokens and gets a reference to the
    /// previous tokens (in that order) when given an index. The index
    /// must be `>= 0` and `< tokens.len()` or this function will result
    /// in UB. Calling this function with an already used index will
    /// result in default tokens returned. This functions behaves
    /// differently, depending on the value of the index to accomodate
    /// the `outside_token`, located at the end of the
    /// `extended_vector` vector. If `index` is 0, the previous token
    /// is the outside token of the extended tokens. Else, it takes
    /// the tokens at index `i` and `i-1`.
    ///
    /// SAFETY: The index must be >= 0 and <= tokens.len()-1, or this
    /// function will result in UB.
    ///
    /// * `tokens`: The tokens. The current and previous tokens are
    ///    extracted from its extended_tokens field.
    /// * `index`: Index specifying the current token. `index-1` is
    ///    used to take the previous token if index!=1.
    #[inline(always)]
    unsafe fn take_out_pair(
        tokens: &'b mut Tokens<'a>,
        index: usize,
    ) -> (Token<'a>, &'b Token<'a>) {
        if index == 0 {
            // The outside token is actually the last token, but is treated as the first one.
            let index_of_outside_token = tokens.extended_tokens.len() - 1;
            let current_token = take(tokens.extended_tokens.get_unchecked_mut(0));
            let previous_token = tokens.extended_tokens.get_unchecked(index_of_outside_token);
            (current_token, previous_token)
        } else {
            let current_token = take(tokens.extended_tokens.get_unchecked_mut(index));
            let previous_token = tokens.extended_tokens.get_unchecked(index - 1);
            (current_token, previous_token)
        }
    }
    fn new(tokens: Tokens<'a>) -> Self {
        let len = tokens.extended_tokens.len();
        Self {
            index: 0,
            tokens: RefCell::new(tokens),
            len,
        }
    }
}

/// The EntitiesIter struct parses the `Tokens` into Entities. The heavy lifting is actually done
/// with the EntitiesIterAdaptor struct.
struct EntitiesIter<'a>(EntitiesIterAdaptor<'a>);

impl<'a> Iterator for EntitiesIter<'a> {
    type Item = Result<Entity<'a>, InvalidToken>;
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let mut res: Option<Option<Result<Entity<'a>, InvalidToken>>> = self.0.next();
        // Removes the Some(None) cases
        while matches!(&res, Some(None)) {
            res = self.0.next();
        }
        // Could be None or Some(Some(..))
        match res {
            Some(Some(result_value)) => Some(result_value),
            None => None,
            Some(None) => unreachable!(),
        }
    }
}

impl<'a> EntitiesIter<'a> {
    fn new(tokens: Tokens<'a>) -> Self {
        let adaptor = EntitiesIterAdaptor::new(tokens);
        EntitiesIter(adaptor)
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Enum of errors wrapping the actual error structs.
pub enum ConversionError<S: AsRef<str>> {
    /// Invalid token encoutered when
    InvalidToken(InvalidToken),
    /// Could not parse the string into a `Prefix`
    ParsingPrefix(ParsingError<S>),
}

// impl ConversionError<&str> {
//     pub(crate) fn to_string(self) -> ConversionError<String> {
//         match self {
//             Self::InvalidToken(t) => Self::InvalidToken(t),
//             Self::ParsingPrefix(ParsingPrefixError(ref_str)) => {
//                 Self::ParsingPrefix(ParsingPrefixError(ref_str.to_string()))
//             }
//         }
//     }
// }

impl<S: AsRef<str>> From<InvalidToken> for ConversionError<S> {
    fn from(value: InvalidToken) -> Self {
        Self::InvalidToken(value)
    }
}

impl<S: AsRef<str>> From<ParsingError<S>> for ConversionError<S> {
    fn from(value: ParsingError<S>) -> Self {
        Self::ParsingPrefix(value)
    }
}

impl<S: AsRef<str>> Display for ConversionError<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidToken(it) => std::fmt::Display::fmt(&it, f),
            Self::ParsingPrefix(pp) => pp.fmt(f),
        }
    }
}

impl<S: AsRef<str> + Debug> Error for ConversionError<S> {}

#[derive(Debug, PartialEq, Clone)]
/// Entites are the unique tokens contained in a sequence. Entities can be built with the
/// TryFromVec trait. This trait allows to collect from a vec
pub(crate) struct Entities<'a>(FlatArray<Entity<'a>>);

impl<'a> Deref for Entities<'a> {
    type Target = FlatArray<Entity<'a>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for Entities<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'a> Entities<'a> {
    fn iter(&'a self) -> Iter<'a, Entity<'a>> {
        self.0.iter()
    }
}

/// This trait mimics the TryFrom trait from the std lib. It is used
/// to *try* to build an Entities structure. It can fail if there is a
/// malformed token in `tokens`.
///
/// * `tokens`: Vector containing the raw tokens.
/// * `scheme`: The scheme type to use (ex: IOB2, BILOU, etc.). The
///    supported scheme are the variant of SchemeType.
/// * `suffix`: Set it to `true` if the Tag is located at the start of
///    the token and set it to `false` if the Tag is located at the
///    end of the token.
/// * ` `: The character used separate the Tag from the Prefix
///    (ex: `I-PER`, where the tag is `PER` and the prefix is `I`)
pub(crate) trait TryFromVecStrict<'a> {
    type Error: Error;
    fn try_from_strict(
        tokens: &'a mut FlatArray<&'a str>,
        scheme: SchemeType,
        suffix: bool,
    ) -> Result<Entities<'a>, Self::Error>;
}

impl<'a> TryFromVecStrict<'a> for Entities<'a> {
    type Error = ConversionError<String>;
    #[inline(always)]
    fn try_from_strict(
        vec_of_tokens_2d: &'a mut FlatArray<&'a str>,
        scheme: SchemeType,
        suffix: bool,
    ) -> Result<Entities<'a>, Self::Error> {
        let len = vec_of_tokens_2d.len();
        let mut tokens = Vec::with_capacity(len);
        let mut_iter = UnsafeCell::new(vec_of_tokens_2d.iter_arrays_mut());
        loop {
            let current = unsafe { &mut *mut_iter.get() };
            let current_next = current.next();
            if current_next.is_none() {
                let res: Result<Vec<Vec<Entity>>, _> = tokens
                    .into_iter()
                    .map(|t| EntitiesIter::new(t).collect())
                    .collect();
                match res {
                    Ok(vec_of_vecs) => {
                        let tok = FlatArray::from(vec_of_vecs);
                        return Ok(Entities::new(tok));
                    }
                    Err(e) => return Err(e.into()),
                }
            } else {
                match Tokens::new(current_next.unwrap(), scheme, suffix) {
                    Ok(t) => tokens.push(t),
                    Err(e) => Err(e)?,
                }
            }
        }
    }
}

impl<'a> Entities<'a> {
    /// Consumes the 2D array of vecs and builds the Entities.
    pub(crate) fn new(entities: FlatArray<Entity<'a>>) -> Self {
        Entities(entities)
    }

    #[inline(always)]
    /// Filters the entities for a given tag name and returns them in a HashSet.
    ///
    /// * `tag_name`: This variable is used to compare the tag of the entity with. Only those whose
    ///   tag is equal to a reference to `tag_name` are added into the returned HashSet.
    pub fn filter<S: AsRef<str>>(&self, tag_name: S) -> AHashSet<&Entity> {
        let tag_name_ref = tag_name.as_ref();
        // NOTE: This one of the most expansive calls:
        AHashSet::from_iter(self.iter().filter(|e| e.tag == tag_name_ref))
    }

    /// Filters the entities for a given tag name and return the number of entities..
    ///
    /// * `tag_name`: This variable is used to compare the tag of the
    ///   entity with. Only those whose tag is equal to a reference to
    ///   `tag_name` are added into the returned HashSet.
    pub fn filter_count<S: AsRef<str>>(&self, tag_name: S) -> usize {
        let tag_name_ref = tag_name.as_ref();
        self.iter().filter(|e| e.tag == tag_name_ref).count()
    }

    pub fn unique_tags(&self) -> AHashSet<&str> {
        // NOTE: This one of the most expansive calls:
        AHashSet::from_iter(self.iter().map(|e| e.tag))
    }
}

#[cfg(test)]
pub(super) mod tests {

    use super::*;
    use enum_iterator::{all, Sequence};
    use quickcheck::{self, TestResult};

    impl<'a> Entity<'a> {
        pub fn as_tuple(&'a self) -> (usize, usize, &'a str) {
            (self.start, self.end, self.tag)
        }
    }

    #[test]
    fn test_entities_try_from() {
        let vec_of_tokens = vec![
            vec!["B-PER", "I-PER", "O", "B-LOC"],
            vec![
                "B-GEO", "I-GEO", "O", "B-GEO", "O", "B-PER", "I-PER", "I-PER", "B-LOC",
            ],
        ];
        let mut vec_of_tokens_2d = FlatArray::new(vec_of_tokens);
        let entities =
            Entities::try_from_strict(&mut vec_of_tokens_2d, SchemeType::IOB2, false).unwrap();
        assert_eq!(
            entities.get_content().to_vec(),
            vec![
                Entity {
                    start: 0,
                    end: 2,
                    tag: "PER"
                },
                Entity {
                    start: 3,
                    end: 4,
                    tag: "LOC"
                },
                Entity {
                    start: 0,
                    end: 2,
                    tag: "GEO"
                },
                Entity {
                    start: 3,
                    end: 4,
                    tag: "GEO"
                },
                Entity {
                    start: 5,
                    end: 8,
                    tag: "PER"
                },
                Entity {
                    start: 8,
                    end: 9,
                    tag: "LOC"
                },
            ]
        );
    }

    #[derive(Debug, PartialEq, Hash, Clone, Sequence, Eq)]
    pub(crate) enum TokensToTest {
        BPER,
        BGEO,
        BLOC,
        O,
    }
    impl From<TokensToTest> for &str {
        fn from(value: TokensToTest) -> Self {
            match value {
                TokensToTest::BPER => "B-PER",
                TokensToTest::BLOC => "B-LOC",
                TokensToTest::BGEO => "B-GEO",
                TokensToTest::O => "O",
            }
        }
    }
    impl quickcheck::Arbitrary for TokensToTest {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            let mut choice_slice: Vec<TokensToTest> = all::<TokensToTest>().collect();
            // Removes the `ALL` prefix
            choice_slice.swap_remove(choice_slice.len() - 1);
            g.choose(choice_slice.as_ref()).unwrap().clone()
        }
    }

    #[test]
    fn test_propertie_entities_try_from() {
        #[allow(non_snake_case)]
        fn propertie_entities_try_from_vecs_strict_IO_only(
            tokens: Vec<Vec<TokensToTest>>,
        ) -> TestResult {
            let mut tok = FlatArray::new(
                tokens
                    .into_iter()
                    .map(|v| v.into_iter().map(|t| t.into()).collect())
                    .collect(),
            );
            let entities = Entities::try_from_strict(&mut tok, SchemeType::IOB2, false).unwrap();
            for entity in entities.iter() {
                let diff = entity.end - entity.start;
                if diff != 1 {
                    return TestResult::failed();
                };
            }
            TestResult::passed()
        }
        let mut qc = quickcheck::QuickCheck::new().tests(2000);
        qc.quickcheck(
            propertie_entities_try_from_vecs_strict_IO_only
                as fn(Vec<Vec<TokensToTest>>) -> TestResult,
        )
    }

    #[test]
    fn test_entities_filter() {
        let mut tokens = build_tokens_vec_str();
        let tok_ref = tokens.as_mut_slice();
        let scheme = SchemeType::IOB2;
        let suffix = false;
        let tokens = Tokens::new(tok_ref, scheme, suffix).unwrap();
        println!("{:?}", tokens);
        let mut tokens = build_tokens_vec_str();
        let tok_ref = tokens.as_mut_slice();
        let scheme = SchemeType::IOB2;
        let suffix = false;
        let tokens = Tokens::new(tok_ref, scheme, suffix).unwrap();
        let entities: Result<Vec<_>, InvalidToken> = EntitiesIter::new(tokens).collect();
        let entities = entities.unwrap();
        let expected = vec![
            Entity {
                start: 0,
                end: 2,
                tag: "PER",
            },
            Entity {
                start: 3,
                end: 4,
                tag: "LOC",
            },
        ];
        assert_eq!(entities, expected);
    }

    // fn build_entities() -> Vec<Entity<'static>> {
    //     let mut tokens = build_tokens_vec_str();
    //     let tok_ref = tokens.as_mut_slice();
    //     let scheme = SchemeType::IOB2;
    //     let   =  ;
    //     let suffix = false;
    //     let tokens = Tokens::new(tok_ref, scheme, suffix,  ).unwrap();
    //     let entities: Result<Vec<_>, InvalidToken> = EntitiesIter::new(tokens).collect();
    //     entities.unwrap()
    // }

    #[test]
    fn test_entity_iter() {
        let mut tokens = build_tokens_vec_str();
        let tok_ref = tokens.as_mut_slice();
        let scheme = SchemeType::IOB2;
        let suffix = false;
        let tokens = Tokens::new(tok_ref, scheme, suffix).unwrap();
        println!("tokens: {:?}", tokens);
        let iter = EntitiesIter(EntitiesIterAdaptor::new(tokens.clone()));
        let wrapped_entities: Result<Vec<_>, InvalidToken> = iter.collect();
        let entities = wrapped_entities.unwrap();
        let expected_entities = vec![
            Entity {
                start: 0,
                end: 2,
                tag: "PER",
            },
            Entity {
                start: 3,
                end: 4,
                tag: "LOC",
            },
        ];
        assert_eq!(expected_entities, entities)
    }

    #[test]
    fn test_entity_adaptor_iterator() {
        let mut tokens = build_tokens_vec_str();
        let tok_ref = tokens.as_mut_slice();
        let scheme = SchemeType::IOB2;
        let suffix = false;
        let tokens = Tokens::new(tok_ref, scheme, suffix).unwrap();
        println!("tokens: {:?}", tokens);
        let mut iter = EntitiesIterAdaptor::new(tokens.clone());
        let first_entity = iter.next().unwrap();
        println!("first entity: {:?}", first_entity);
        assert!(first_entity.is_some());
        let second_entity = iter.next().unwrap();
        println!("second entity: {:?}", second_entity);
        assert!(second_entity.is_none());
        let third_entity = iter.next().unwrap();
        println!("third entity: {:?}", third_entity);
        assert!(third_entity.is_some());
        // let forth_entity = iter.next().unwrap();
        // println!("forth entity: {:?}", forth_entity);
        // assert!(forth_entity.is_none());
        let iteration_has_ended = iter.next().is_none();
        assert!(iteration_has_ended);
    }

    #[test]
    fn test_is_start() {
        let mut tokens = build_tokens_vec_str();
        let tok_ref = tokens.as_mut_slice();
        let scheme = SchemeType::IOB2;
        let suffix = false;
        let tokens = Tokens::new(tok_ref, scheme, suffix).unwrap();
        dbg!(tokens.clone());
        let first_token = tokens.extended_tokens.first().unwrap();
        let second_token = tokens.extended_tokens.get(1).unwrap();
        assert!(first_token.is_start(second_token.inner()));
        let outside_token = tokens.extended_tokens.last().unwrap();
        assert!(first_token.is_start(outside_token.inner()));
    }
    #[test]
    fn test_tokens_is_end() {
        let mut tokens = build_tokens_vec_str();
        let tok_ref = tokens.as_mut_slice();
        let scheme = SchemeType::IOB2;
        let suffix = false;
        let tokens = Tokens::new(tok_ref, scheme, suffix).unwrap();
        let is_end_of_chunk = tokens.is_end(2);
        dbg!(tokens.clone());
        // let first_non_outside_token = &tokens.extended_tokens.get(1).unwrap();
        // let second_non_outside_token = &tokens.extended_tokens.get(2).unwrap();
        assert!(is_end_of_chunk);
        let is_end_of_chunk = tokens.is_end(3);
        assert!(!is_end_of_chunk)
    }

    #[test]
    fn test_innertoken_is_end() {
        let mut tokens = build_tokens_vec_str();
        let tok_ref = tokens.as_mut_slice();
        let scheme = SchemeType::IOB2;
        let suffix = false;
        let tokens = Tokens::new(tok_ref, scheme, suffix).unwrap();
        let first_non_outside_token = tokens.extended_tokens.first().unwrap();
        let second_non_outside_token = tokens.extended_tokens.get(1).unwrap();
        let third_non_outside_token = tokens.extended_tokens.get(2).unwrap();
        let is_end = second_non_outside_token.is_end(first_non_outside_token.inner());
        assert!(!is_end);
        let is_end = third_non_outside_token.is_end(first_non_outside_token.inner());
        assert!(is_end)
    }

    #[test]
    fn test_token_is_start() {
        let mut tokens = build_tokens_vec_str();
        let tok_ref = tokens.as_mut_slice();
        let scheme = SchemeType::IOB2;
        let suffix = false;
        let tokens = Tokens::new(tok_ref, scheme, suffix).unwrap();
        println!("{:?}", tokens);
        println!("{:?}", tokens.extended_tokens());
        let prev = tokens.extended_tokens().first().unwrap();
        let is_start = tokens
            .extended_tokens()
            .get(1)
            .unwrap()
            .is_start(prev.inner());
        assert!(!is_start)
    }
    #[test]
    fn test_forward_method() {
        let mut tokens = build_tokens_vec_str();
        let tok_ref = tokens.as_mut_slice();
        let scheme = SchemeType::IOB2;
        let suffix = false;
        let tokens = Tokens::new(tok_ref, scheme, suffix).unwrap();
        println!("{:?}", &tokens);
        let end = tokens.forward(1, tokens.extended_tokens.first().unwrap());
        let expected_end = 2;
        assert_eq!(end, expected_end)
    }
    #[test]
    fn test_new_tokens() {
        let mut tokens = build_tokens_vec_str();
        let tok_ref = tokens.as_mut_slice();
        let scheme = SchemeType::IOB2;
        let suffix = false;
        let tokens = Tokens::new(tok_ref, scheme, suffix).unwrap();
        println!("{:?}", tokens);
        assert_eq!(tokens.extended_tokens.len(), 5);
    }

    #[test]
    fn test_unique_tags() {
        let mut sequences = FlatArray::new(vec![build_str_vec(), build_str_vec_diff()]);
        let entities = Entities::try_from_strict(&mut sequences, SchemeType::IOB2, false).unwrap();
        let actual_unique_tags = entities.unique_tags();
        let expected_unique_tags: AHashSet<&str> = AHashSet::from_iter(vec!["PER", "LOC", "GEO"]);
        assert_eq!(actual_unique_tags, expected_unique_tags);
    }

    #[test]
    fn test_get_entities_lenient() {
        let tokens = vec!["B-PER", "I-PER", "O", "B-LOC"];
        let seq = FlatArray::new(vec![tokens.clone()]);
        let actual = get_entities_lenient(&seq, false).unwrap();
        let entities = vec![Entity::new(0, 2, "PER"), Entity::new(3, 4, "LOC")];
        let expected_tokens = entities.into_boxed_slice();
        let expected_indices = vec![0, tokens.len()];
        let expected_inner = FlatArray::from_raw(expected_tokens, expected_indices);
        let expected = Entities(expected_inner);
        assert_eq!(actual, expected)
    }

    #[test]
    fn test_get_entities_lenient_prefix() {
        let y_true = vec![vec![
            "O", "O", "O", "B-MISC", "I-MISC", "I-MISC", "O", "B-PER", "I-PER",
        ]];
        let y_true = FlatArray::new(y_true);
        let actual = get_entities_lenient(&y_true, false).unwrap();
        assert_eq!(
            actual
                .0
                .get_content()
                .iter()
                .map(|e| e.as_tuple())
                .collect::<Vec<_>>(),
            Vec::from([(3, 6, "MISC"), (7, 9, "PER")])
        )
    }

    #[test]
    fn test_get_entities_lenient_suffix() {
        let y_true = vec![vec![
            "O", "O", "O", "MISC-B", "MISC-I", "MISC-I", "O", "PER-B", "PER-I",
        ]];
        let y_true = FlatArray::new(y_true);
        let actual = get_entities_lenient(&y_true, true).unwrap();
        assert_eq!(
            actual
                .0
                .get_content()
                .iter()
                .map(|e| e.as_tuple())
                .collect::<Vec<_>>(),
            Vec::from([(3, 6, "MISC"), (7, 9, "PER")])
        )
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_get_entities_with_only_IOB() {
        let y_true = vec![vec!["O", "O", "O", "B", "I", "I", "O"], vec!["B", "I", "O"]];
        let y_true = FlatArray::new(y_true);
        let actual = get_entities_lenient(&y_true, true).unwrap();
        assert_eq!(
            actual
                .0
                .get_content()
                .iter()
                .map(|e| e.as_tuple())
                .collect::<Vec<_>>(),
            Vec::from([(3, 6, "_"), (0, 2, "_")])
        )
    }

    #[allow(non_snake_case)]
    #[test]
    fn test_LenientChunkIterator() {
        let tokens = build_str_vec();
        let iter = LenientChunkIter::new(tokens.as_ref(), false);
        let actual = iter.collect::<Vec<_>>();
        let expected: Vec<Result<Entity, ParsingError<String>>> =
            vec![Ok(Entity::new(0, 2, "PER")), Ok(Entity::new(3, 4, "LOC"))];
        assert_eq!(actual, expected)
    }

    #[test]
    fn test_get_entities_lenient_2() {
        let seq = vec![vec![
            "O", "O", "O", "B-MISC", "I-MISC", "I-MISC", "O", "B-PER", "I-PER",
        ]];
        let binding = &seq.into();
        let binding2 = get_entities_lenient(binding, false).unwrap();
        let actual = binding2.0.iter().map(|e| e.as_tuple()).collect::<Vec<_>>();
        let expected: Vec<(usize, usize, &str)> = vec![(3, 6, "MISC"), (7, 9, "PER")];
        assert_eq!(expected, actual)
    }

    #[test]
    fn test_get_entities_lenient_with_suffix() {
        let seq = vec![vec![
            "O", "O", "O", "MISC-B", "MISC-I", "MISC-I", "O", "PER-B", "PER-I",
        ]];
        let binding = &seq.into();
        let binding2 = get_entities_lenient(binding, true).unwrap();
        let actual = binding2.0.iter().map(|e| e.as_tuple()).collect::<Vec<_>>();
        let expected: Vec<(usize, usize, &str)> = vec![(3, 6, "MISC"), (7, 9, "PER")];
        assert_eq!(expected, actual)
    }

    fn build_tokens_vec_str() -> Vec<&'static str> {
        vec!["B-PER", "I-PER", "O", "B-LOC"]
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
