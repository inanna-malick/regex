/*!
Defines a translator that converts an `Ast` to an `Hir`.
*/

use core::cell::Cell;

use alloc::{boxed::Box, string::ToString, vec, vec::Vec};

use crate::{
    ast::{
        self,
        visitor::{project_ast, AstFrame},
        Ast, Span,
    },
    either::Either,
    hir::{self, Error, ErrorKind, Hir, HirKind},
    unicode::{self, ClassQuery},
};

use recursion::{
    expand_and_collapse, try_expand_and_collapse, MappableFrame, PartiallyApplied,
};

type Result<T> = core::result::Result<T, Error>;

/// A builder for constructing an AST->HIR translator.
#[derive(Clone, Debug)]
pub struct TranslatorBuilder {
    utf8: bool,
    line_terminator: u8,
    flags: Flags,
}

impl Default for TranslatorBuilder {
    fn default() -> TranslatorBuilder {
        TranslatorBuilder::new()
    }
}

impl TranslatorBuilder {
    /// Create a new translator builder with a default configuration.
    pub fn new() -> TranslatorBuilder {
        TranslatorBuilder {
            utf8: true,
            line_terminator: b'\n',
            flags: Flags::default(),
        }
    }

    /// Build a translator using the current configuration.
    pub fn build(&self) -> Translator {
        Translator {
            flags: Cell::new(self.flags),
            utf8: self.utf8,
            line_terminator: self.line_terminator,
        }
    }

    /// When disabled, translation will permit the construction of a regular
    /// expression that may match invalid UTF-8.
    ///
    /// When enabled (the default), the translator is guaranteed to produce an
    /// expression that, for non-empty matches, will only ever produce spans
    /// that are entirely valid UTF-8 (otherwise, the translator will return an
    /// error).
    ///
    /// Perhaps surprisingly, when UTF-8 is enabled, an empty regex or even
    /// a negated ASCII word boundary (uttered as `(?-u:\B)` in the concrete
    /// syntax) will be allowed even though they can produce matches that split
    /// a UTF-8 encoded codepoint. This only applies to zero-width or "empty"
    /// matches, and it is expected that the regex engine itself must handle
    /// these cases if necessary (perhaps by suppressing any zero-width matches
    /// that split a codepoint).
    pub fn utf8(&mut self, yes: bool) -> &mut TranslatorBuilder {
        self.utf8 = yes;
        self
    }

    /// Sets the line terminator for use with `(?u-s:.)` and `(?-us:.)`.
    ///
    /// Namely, instead of `.` (by default) matching everything except for `\n`,
    /// this will cause `.` to match everything except for the byte given.
    ///
    /// If `.` is used in a context where Unicode mode is enabled and this byte
    /// isn't ASCII, then an error will be returned. When Unicode mode is
    /// disabled, then any byte is permitted, but will return an error if UTF-8
    /// mode is enabled and it is a non-ASCII byte.
    ///
    /// In short, any ASCII value for a line terminator is always okay. But a
    /// non-ASCII byte might result in an error depending on whether Unicode
    /// mode or UTF-8 mode are enabled.
    ///
    /// Note that if `R` mode is enabled then it always takes precedence and
    /// the line terminator will be treated as `\r` and `\n` simultaneously.
    ///
    /// Note also that this *doesn't* impact the look-around assertions
    /// `(?m:^)` and `(?m:$)`. That's usually controlled by additional
    /// configuration in the regex engine itself.
    pub fn line_terminator(&mut self, byte: u8) -> &mut TranslatorBuilder {
        self.line_terminator = byte;
        self
    }

    /// Enable or disable the case insensitive flag (`i`) by default.
    pub fn case_insensitive(&mut self, yes: bool) -> &mut TranslatorBuilder {
        self.flags.case_insensitive = if yes { Some(true) } else { None };
        self
    }

    /// Enable or disable the multi-line matching flag (`m`) by default.
    pub fn multi_line(&mut self, yes: bool) -> &mut TranslatorBuilder {
        self.flags.multi_line = if yes { Some(true) } else { None };
        self
    }

    /// Enable or disable the "dot matches any character" flag (`s`) by
    /// default.
    pub fn dot_matches_new_line(
        &mut self,
        yes: bool,
    ) -> &mut TranslatorBuilder {
        self.flags.dot_matches_new_line = if yes { Some(true) } else { None };
        self
    }

    /// Enable or disable the CRLF mode flag (`R`) by default.
    pub fn crlf(&mut self, yes: bool) -> &mut TranslatorBuilder {
        self.flags.crlf = if yes { Some(true) } else { None };
        self
    }

    /// Enable or disable the "swap greed" flag (`U`) by default.
    pub fn swap_greed(&mut self, yes: bool) -> &mut TranslatorBuilder {
        self.flags.swap_greed = if yes { Some(true) } else { None };
        self
    }

    /// Enable or disable the Unicode flag (`u`) by default.
    pub fn unicode(&mut self, yes: bool) -> &mut TranslatorBuilder {
        self.flags.unicode = if yes { None } else { Some(false) };
        self
    }
}

/// A translator maps abstract syntax to a high level intermediate
/// representation.
///
/// A translator may be benefit from reuse. That is, a translator can translate
/// many abstract syntax trees.
///
/// A `Translator` can be configured in more detail via a
/// [`TranslatorBuilder`].
#[derive(Clone, Debug)]
pub struct Translator {
    /// The current flag settings.
    flags: Cell<Flags>,
    /// Whether we're allowed to produce HIR that can match arbitrary bytes.
    utf8: bool,
    /// The line terminator to use for `.`.
    line_terminator: u8,
}

impl Translator {
    /// Create a new translator using the default configuration.
    pub fn new() -> Translator {
        TranslatorBuilder::new().build()
    }

    /// Compute the flags that would be active after traversing an AST subtree.
    ///
    /// This mimics the original visitor's behavior where flags set via `(?i)` etc.
    /// persist across sibling nodes in the traversal order.
    ///
    /// Uses a catamorphism that collapses to `FlagOp` (a representation of the
    /// flag transformation), then applies it to the input flags.
    fn compute_exit_flags(ast: &Ast, flags: Flags) -> Flags {
        let op = expand_and_collapse::<AstFrame<PartiallyApplied>, _, _>(
            ast,
            project_ast,
            |frame| match frame {
                // Leaf nodes: identity transformation
                AstFrame::Empty(_)
                | AstFrame::Literal(_)
                | AstFrame::Dot(_)
                | AstFrame::Assertion(_)
                | AstFrame::ClassUnicode(_)
                | AstFrame::ClassPerl(_)
                | AstFrame::ClassBracketed(_) => FlagOp::Identity,

                // Flags node: merge with parsed flags
                AstFrame::Flags(set_flags) => {
                    FlagOp::SetFlags(&set_flags.flags)
                }

                // Groups restore flags on exit, so identity
                AstFrame::Group { .. } => FlagOp::Identity,

                // Repetition: delegate to child's transformation
                AstFrame::Repetition { child, .. } => child,

                // Concat/Alternation: compose children left-to-right
                AstFrame::Concat { children, .. }
                | AstFrame::Alternation { children, .. } => {
                    FlagOp::Compose(children)
                }
            },
        );
        op.apply(flags)
    }

    /// Translate the given abstract syntax tree (AST) into a high level
    /// intermediate representation (HIR).
    ///
    /// If there was a problem doing the translation, then an HIR-specific
    /// error is returned.
    ///
    /// The original pattern string used to produce the `Ast` *must* also be
    /// provided. The translator does not use the pattern string during any
    /// correct translation, but is used for error reporting.
    pub fn translate(&mut self, pattern: &str, ast: &Ast) -> Result<Hir> {
        let ti = TranslatorI::new(self, pattern);
        let initial_flags = ti.trans().flags.get();

        try_expand_and_collapse::<FrameWithFlags<PartiallyApplied>, _, _, _>(
            (ast, initial_flags),
            |(node, flags)| {
                // Expand: project the AST and compute child flags
                // Handle flag propagation specially for Concat and Alternation
                match node {
                    Ast::Concat(c) => {
                        // For concatenation, flags from Ast::Flags nodes affect subsequent siblings
                        let mut current_flags = flags;
                        let children: Vec<_> = c
                            .asts
                            .iter()
                            .map(|child| {
                                // If this child is a Flags node, update flags for subsequent siblings
                                if let Ast::Flags(ref f) = child {
                                    let mut new_flags =
                                        Flags::from_ast(&f.flags);
                                    new_flags.merge(&current_flags);
                                    current_flags = new_flags;
                                }
                                (child, current_flags)
                            })
                            .collect();
                        Ok(FrameWithFlags {
                            flags,
                            frame: AstFrame::Concat { span: c.span, children },
                        })
                    }
                    Ast::Group(g) => {
                        // For groups with flags, update flags for the child
                        let child_flags = match &g.kind {
                            ast::GroupKind::NonCapturing(ast_flags) => {
                                let mut new_flags = Flags::from_ast(ast_flags);
                                new_flags.merge(&flags);
                                new_flags
                            }
                            _ => flags,
                        };
                        Ok(FrameWithFlags {
                            flags,
                            frame: AstFrame::Group {
                                span: g.span,
                                kind: &g.kind,
                                child: (&*g.ast, child_flags),
                            },
                        })
                    }
                    Ast::Alternation(a) => {
                        // For alternation, flags from Flags nodes at the start of earlier
                        // branches affect later branches. This matches the original visitor
                        // behavior where set_flags() persists across alternation branches.
                        let mut current_flags = flags;
                        let children: Vec<_> = a
                            .asts
                            .iter()
                            .map(|child| {
                                let child_flags = current_flags;
                                // Check if this branch starts with flags (directly or in a Concat)
                                current_flags = Self::compute_exit_flags(
                                    child,
                                    current_flags,
                                );
                                (child, child_flags)
                            })
                            .collect();
                        Ok(FrameWithFlags {
                            flags,
                            frame: AstFrame::Alternation {
                                span: a.span,
                                children,
                            },
                        })
                    }
                    _ => {
                        // For other nodes, just project and pass flags through
                        let frame = project_ast(node);
                        Ok(FrameWithFlags {
                            flags,
                            frame: AstFrame::map_frame(frame, |child| {
                                (child, flags)
                            }),
                        })
                    }
                }
            },
            |fwf| ti.collapse_frame(fwf),
        )
    }
}

/// A frame wrapper that carries the current node's flags through collapse.
#[derive(Clone, Debug)]
struct FrameWithFlags<'a, A> {
    flags: Flags,
    frame: AstFrame<'a, A>,
}

impl<'a> MappableFrame for FrameWithFlags<'a, PartiallyApplied> {
    type Frame<X> = FrameWithFlags<'a, X>;

    fn map_frame<A, B>(
        input: Self::Frame<A>,
        f: impl FnMut(A) -> B,
    ) -> Self::Frame<B> {
        FrameWithFlags {
            flags: input.flags,
            frame: AstFrame::map_frame(input.frame, f),
        }
    }
}

/// The internal implementation of a translator.
///
/// This type is responsible for carrying around the original pattern string,
/// which is not tied to the internal state of a translator.
///
/// A TranslatorI exists for the time it takes to translate a single Ast.
#[derive(Clone, Debug)]
struct TranslatorI<'t, 'p> {
    trans: &'t Translator,
    pattern: &'p str,
}

impl<'t, 'p> TranslatorI<'t, 'p> {
    /// Build a new internal translator.
    fn new(trans: &'t Translator, pattern: &'p str) -> TranslatorI<'t, 'p> {
        TranslatorI { trans, pattern }
    }

    /// Return a reference to the underlying translator.
    fn trans(&self) -> &Translator {
        &self.trans
    }

    /// Create a new error with the given span and error type.
    fn error(&self, span: Span, kind: ErrorKind) -> Error {
        Error { kind, pattern: self.pattern.to_string(), span }
    }

    /// Return a copy of the active flags.
    fn flags(&self) -> Flags {
        self.trans().flags.get()
    }

    /// Collapse a FrameWithFlags into an Hir.
    ///
    /// This is the algebra for the catamorphism. Each frame carries its own flags
    /// and contains children that have already been collapsed to Hir.
    fn collapse_frame(&self, fwf: FrameWithFlags<Hir>) -> Result<Hir> {
        let FrameWithFlags { flags, frame } = fwf;
        // Set flags so helper methods see the correct context
        self.trans().flags.set(flags);

        match frame {
            AstFrame::Empty(_) => Ok(Hir::empty()),
            AstFrame::Flags(_x) => {
                // Flags in the AST are generally considered directives and
                // not actual sub-expressions. However, they can be used in
                // the concrete syntax like `((?i))`, and we need some kind of
                // indication of an expression there, and Empty is the correct
                // choice.
                Ok(Hir::empty())
            }
            AstFrame::Literal(ref x) => self.hir_literal(x),
            AstFrame::Dot(span) => self.hir_dot(span),
            AstFrame::Assertion(ref x) => self.hir_assertion(x),
            AstFrame::ClassPerl(ref x) => {
                if self.flags().unicode() {
                    let cls = self.hir_perl_unicode_class(x)?;
                    let hcls = hir::Class::Unicode(cls);
                    Ok(Hir::class(hcls))
                } else {
                    let cls = self.hir_perl_byte_class(x)?;
                    let hcls = hir::Class::Bytes(cls);
                    Ok(Hir::class(hcls))
                }
            }
            AstFrame::ClassUnicode(ref x) => {
                let cls = hir::Class::Unicode(self.hir_unicode_class(x)?);
                Ok(Hir::class(cls))
            }
            AstFrame::ClassBracketed(ref ast) => self.hir_class_bracketed(ast),
            AstFrame::Repetition { span: _, ref op, greedy, child: expr } => {
                Ok(self.hir_repetition_impl(op, greedy, expr))
            }
            AstFrame::Group { span: _, ref kind, child: expr } => {
                Ok(self.hir_group(kind, expr))
            }
            AstFrame::Concat { span: _, children } => {
                let exprs: Vec<Hir> = children
                    .into_iter()
                    .filter(|expr| !matches!(*expr.kind(), HirKind::Empty))
                    .collect();
                Ok(Hir::concat(exprs))
            }
            AstFrame::Alternation { span: _, children } => {
                Ok(Hir::alternation(children))
            }
        }
    }

    /// Translate a literal AST node to HIR.
    fn hir_literal(&self, lit: &ast::Literal) -> Result<Hir> {
        match self.ast_literal_to_scalar(lit)? {
            Either::Right(byte) => Ok(Hir::literal([byte])),
            Either::Left(ch) => match self.case_fold_char(lit.span, ch)? {
                None => {
                    let mut buf = [0u8; 4];
                    let s = ch.encode_utf8(&mut buf);
                    Ok(Hir::literal(s.as_bytes().to_vec()))
                }
                Some(expr) => Ok(expr),
            },
        }
    }

    /// Translate a group AST node to HIR.
    fn hir_group(&self, kind: &ast::GroupKind, expr: Hir) -> Hir {
        let (index, name) = match kind {
            ast::GroupKind::CaptureIndex(index) => (*index, None),
            ast::GroupKind::CaptureName { ref name, .. } => {
                (name.index, Some(name.name.clone().into_boxed_str()))
            }
            // The HIR doesn't need to use non-capturing groups, since the way
            // in which the data type is defined handles this automatically.
            ast::GroupKind::NonCapturing(_) => return expr,
        };
        Hir::capture(hir::Capture { index, name, sub: Box::new(expr) })
    }

    /// Translate a repetition AST node to HIR.
    fn hir_repetition_impl(
        &self,
        op: &ast::RepetitionOp,
        greedy: bool,
        expr: Hir,
    ) -> Hir {
        let (min, max) = match op.kind {
            ast::RepetitionKind::ZeroOrOne => (0, Some(1)),
            ast::RepetitionKind::ZeroOrMore => (0, None),
            ast::RepetitionKind::OneOrMore => (1, None),
            ast::RepetitionKind::Range(ast::RepetitionRange::Exactly(m)) => {
                (m, Some(m))
            }
            ast::RepetitionKind::Range(ast::RepetitionRange::AtLeast(m)) => {
                (m, None)
            }
            ast::RepetitionKind::Range(ast::RepetitionRange::Bounded(
                m,
                n,
            )) => (m, Some(n)),
        };
        let greedy = if self.flags().swap_greed() { !greedy } else { greedy };
        Hir::repetition(hir::Repetition {
            min,
            max,
            greedy,
            sub: Box::new(expr),
        })
    }

    /// Translate a bracketed character class to HIR.
    fn hir_class_bracketed(&self, ast: &ast::ClassBracketed) -> Result<Hir> {
        if self.flags().unicode() {
            let mut cls = self.hir_class_set_unicode(&ast.kind)?;
            self.unicode_fold_and_negate(&ast.span, ast.negated, &mut cls)?;
            Ok(Hir::class(hir::Class::Unicode(cls)))
        } else {
            let mut cls = self.hir_class_set_bytes(&ast.kind)?;
            self.bytes_fold_and_negate(&ast.span, ast.negated, &mut cls)?;
            Ok(Hir::class(hir::Class::Bytes(cls)))
        }
    }

    /// Translate a ClassSet to a Unicode class.
    fn hir_class_set_unicode(
        &self,
        set: &ast::ClassSet,
    ) -> Result<hir::ClassUnicode> {
        match set {
            ast::ClassSet::Item(item) => self.hir_class_set_item_unicode(item),
            ast::ClassSet::BinaryOp(op) => {
                self.hir_class_binary_op_unicode(op)
            }
        }
    }

    /// Translate a ClassSet to a byte class.
    fn hir_class_set_bytes(
        &self,
        set: &ast::ClassSet,
    ) -> Result<hir::ClassBytes> {
        match set {
            ast::ClassSet::Item(item) => self.hir_class_set_item_bytes(item),
            ast::ClassSet::BinaryOp(op) => self.hir_class_binary_op_bytes(op),
        }
    }

    /// Translate a ClassSetItem to a Unicode class.
    fn hir_class_set_item_unicode(
        &self,
        item: &ast::ClassSetItem,
    ) -> Result<hir::ClassUnicode> {
        match item {
            ast::ClassSetItem::Empty(_) => Ok(hir::ClassUnicode::empty()),
            ast::ClassSetItem::Literal(x) => {
                let mut cls = hir::ClassUnicode::empty();
                cls.push(hir::ClassUnicodeRange::new(x.c, x.c));
                Ok(cls)
            }
            ast::ClassSetItem::Range(x) => {
                let mut cls = hir::ClassUnicode::empty();
                cls.push(hir::ClassUnicodeRange::new(x.start.c, x.end.c));
                Ok(cls)
            }
            ast::ClassSetItem::Ascii(x) => self.hir_ascii_unicode_class(x),
            ast::ClassSetItem::Unicode(x) => self.hir_unicode_class(x),
            ast::ClassSetItem::Perl(x) => self.hir_perl_unicode_class(x),
            ast::ClassSetItem::Bracketed(x) => {
                let mut cls = self.hir_class_set_unicode(&x.kind)?;
                self.unicode_fold_and_negate(&x.span, x.negated, &mut cls)?;
                Ok(cls)
            }
            ast::ClassSetItem::Union(x) => {
                let mut cls = hir::ClassUnicode::empty();
                for item in &x.items {
                    let item_cls = self.hir_class_set_item_unicode(item)?;
                    cls.union(&item_cls);
                }
                Ok(cls)
            }
        }
    }

    /// Translate a ClassSetItem to a byte class.
    fn hir_class_set_item_bytes(
        &self,
        item: &ast::ClassSetItem,
    ) -> Result<hir::ClassBytes> {
        match item {
            ast::ClassSetItem::Empty(_) => Ok(hir::ClassBytes::empty()),
            ast::ClassSetItem::Literal(x) => {
                let mut cls = hir::ClassBytes::empty();
                let byte = self.class_literal_byte(x)?;
                cls.push(hir::ClassBytesRange::new(byte, byte));
                Ok(cls)
            }
            ast::ClassSetItem::Range(x) => {
                let mut cls = hir::ClassBytes::empty();
                let start = self.class_literal_byte(&x.start)?;
                let end = self.class_literal_byte(&x.end)?;
                cls.push(hir::ClassBytesRange::new(start, end));
                Ok(cls)
            }
            ast::ClassSetItem::Ascii(x) => self.hir_ascii_byte_class(x),
            ast::ClassSetItem::Unicode(x) => {
                Err(self.error(x.span, ErrorKind::UnicodeNotAllowed))
            }
            ast::ClassSetItem::Perl(x) => self.hir_perl_byte_class(x),
            ast::ClassSetItem::Bracketed(x) => {
                let mut cls = self.hir_class_set_bytes(&x.kind)?;
                self.bytes_fold_and_negate(&x.span, x.negated, &mut cls)?;
                Ok(cls)
            }
            ast::ClassSetItem::Union(x) => {
                let mut cls = hir::ClassBytes::empty();
                for item in &x.items {
                    let item_cls = self.hir_class_set_item_bytes(item)?;
                    cls.union(&item_cls);
                }
                Ok(cls)
            }
        }
    }

    /// Translate a binary class set operation to a Unicode class.
    fn hir_class_binary_op_unicode(
        &self,
        op: &ast::ClassSetBinaryOp,
    ) -> Result<hir::ClassUnicode> {
        use crate::ast::ClassSetBinaryOpKind::*;

        let mut lhs = self.hir_class_set_unicode(&op.lhs)?;
        let mut rhs = self.hir_class_set_unicode(&op.rhs)?;

        if self.flags().case_insensitive() {
            lhs.try_case_fold_simple().map_err(|_| {
                self.error(
                    op.lhs.span().clone(),
                    ErrorKind::UnicodeCaseUnavailable,
                )
            })?;
            rhs.try_case_fold_simple().map_err(|_| {
                self.error(
                    op.rhs.span().clone(),
                    ErrorKind::UnicodeCaseUnavailable,
                )
            })?;
        }

        match op.kind {
            Intersection => lhs.intersect(&rhs),
            Difference => lhs.difference(&rhs),
            SymmetricDifference => lhs.symmetric_difference(&rhs),
        }
        Ok(lhs)
    }

    /// Translate a binary class set operation to a byte class.
    fn hir_class_binary_op_bytes(
        &self,
        op: &ast::ClassSetBinaryOp,
    ) -> Result<hir::ClassBytes> {
        use crate::ast::ClassSetBinaryOpKind::*;

        let mut lhs = self.hir_class_set_bytes(&op.lhs)?;
        let mut rhs = self.hir_class_set_bytes(&op.rhs)?;

        if self.flags().case_insensitive() {
            lhs.case_fold_simple();
            rhs.case_fold_simple();
        }

        match op.kind {
            Intersection => lhs.intersect(&rhs),
            Difference => lhs.difference(&rhs),
            SymmetricDifference => lhs.symmetric_difference(&rhs),
        }
        Ok(lhs)
    }

    /// Convert an Ast literal to its scalar representation.
    ///
    /// When Unicode mode is enabled, then this always succeeds and returns a
    /// `char` (Unicode scalar value).
    ///
    /// When Unicode mode is disabled, then a `char` will still be returned
    /// whenever possible. A byte is returned only when invalid UTF-8 is
    /// allowed and when the byte is not ASCII. Otherwise, a non-ASCII byte
    /// will result in an error when invalid UTF-8 is not allowed.
    fn ast_literal_to_scalar(
        &self,
        lit: &ast::Literal,
    ) -> Result<Either<char, u8>> {
        if self.flags().unicode() {
            return Ok(Either::Left(lit.c));
        }
        let byte = match lit.byte() {
            None => return Ok(Either::Left(lit.c)),
            Some(byte) => byte,
        };
        if byte <= 0x7F {
            return Ok(Either::Left(char::try_from(byte).unwrap()));
        }
        if self.trans().utf8 {
            return Err(self.error(lit.span, ErrorKind::InvalidUtf8));
        }
        Ok(Either::Right(byte))
    }

    fn case_fold_char(&self, span: Span, c: char) -> Result<Option<Hir>> {
        if !self.flags().case_insensitive() {
            return Ok(None);
        }
        if self.flags().unicode() {
            // If case folding won't do anything, then don't bother trying.
            let map = unicode::SimpleCaseFolder::new()
                .map(|f| f.overlaps(c, c))
                .map_err(|_| {
                    self.error(span, ErrorKind::UnicodeCaseUnavailable)
                })?;
            if !map {
                return Ok(None);
            }
            let mut cls =
                hir::ClassUnicode::new(vec![hir::ClassUnicodeRange::new(
                    c, c,
                )]);
            cls.try_case_fold_simple().map_err(|_| {
                self.error(span, ErrorKind::UnicodeCaseUnavailable)
            })?;
            Ok(Some(Hir::class(hir::Class::Unicode(cls))))
        } else {
            if !c.is_ascii() {
                return Ok(None);
            }
            // If case folding won't do anything, then don't bother trying.
            match c {
                'A'..='Z' | 'a'..='z' => {}
                _ => return Ok(None),
            }
            let mut cls =
                hir::ClassBytes::new(vec![hir::ClassBytesRange::new(
                    // OK because 'c.len_utf8() == 1' which in turn implies
                    // that 'c' is ASCII.
                    u8::try_from(c).unwrap(),
                    u8::try_from(c).unwrap(),
                )]);
            cls.case_fold_simple();
            Ok(Some(Hir::class(hir::Class::Bytes(cls))))
        }
    }

    fn hir_dot(&self, span: Span) -> Result<Hir> {
        let (utf8, lineterm, flags) =
            (self.trans().utf8, self.trans().line_terminator, self.flags());
        if utf8 && (!flags.unicode() || !lineterm.is_ascii()) {
            return Err(self.error(span, ErrorKind::InvalidUtf8));
        }
        let dot = if flags.dot_matches_new_line() {
            if flags.unicode() {
                hir::Dot::AnyChar
            } else {
                hir::Dot::AnyByte
            }
        } else {
            if flags.unicode() {
                if flags.crlf() {
                    hir::Dot::AnyCharExceptCRLF
                } else {
                    if !lineterm.is_ascii() {
                        return Err(
                            self.error(span, ErrorKind::InvalidLineTerminator)
                        );
                    }
                    hir::Dot::AnyCharExcept(char::from(lineterm))
                }
            } else {
                if flags.crlf() {
                    hir::Dot::AnyByteExceptCRLF
                } else {
                    hir::Dot::AnyByteExcept(lineterm)
                }
            }
        };
        Ok(Hir::dot(dot))
    }

    fn hir_assertion(&self, asst: &ast::Assertion) -> Result<Hir> {
        let unicode = self.flags().unicode();
        let multi_line = self.flags().multi_line();
        let crlf = self.flags().crlf();
        Ok(match asst.kind {
            ast::AssertionKind::StartLine => Hir::look(if multi_line {
                if crlf {
                    hir::Look::StartCRLF
                } else {
                    hir::Look::StartLF
                }
            } else {
                hir::Look::Start
            }),
            ast::AssertionKind::EndLine => Hir::look(if multi_line {
                if crlf {
                    hir::Look::EndCRLF
                } else {
                    hir::Look::EndLF
                }
            } else {
                hir::Look::End
            }),
            ast::AssertionKind::StartText => Hir::look(hir::Look::Start),
            ast::AssertionKind::EndText => Hir::look(hir::Look::End),
            ast::AssertionKind::WordBoundary => Hir::look(if unicode {
                hir::Look::WordUnicode
            } else {
                hir::Look::WordAscii
            }),
            ast::AssertionKind::NotWordBoundary => Hir::look(if unicode {
                hir::Look::WordUnicodeNegate
            } else {
                hir::Look::WordAsciiNegate
            }),
            ast::AssertionKind::WordBoundaryStart
            | ast::AssertionKind::WordBoundaryStartAngle => {
                Hir::look(if unicode {
                    hir::Look::WordStartUnicode
                } else {
                    hir::Look::WordStartAscii
                })
            }
            ast::AssertionKind::WordBoundaryEnd
            | ast::AssertionKind::WordBoundaryEndAngle => {
                Hir::look(if unicode {
                    hir::Look::WordEndUnicode
                } else {
                    hir::Look::WordEndAscii
                })
            }
            ast::AssertionKind::WordBoundaryStartHalf => {
                Hir::look(if unicode {
                    hir::Look::WordStartHalfUnicode
                } else {
                    hir::Look::WordStartHalfAscii
                })
            }
            ast::AssertionKind::WordBoundaryEndHalf => Hir::look(if unicode {
                hir::Look::WordEndHalfUnicode
            } else {
                hir::Look::WordEndHalfAscii
            }),
        })
    }

    fn hir_unicode_class(
        &self,
        ast_class: &ast::ClassUnicode,
    ) -> Result<hir::ClassUnicode> {
        use crate::ast::ClassUnicodeKind::*;

        if !self.flags().unicode() {
            return Err(
                self.error(ast_class.span, ErrorKind::UnicodeNotAllowed)
            );
        }
        let query = match ast_class.kind {
            OneLetter(name) => ClassQuery::OneLetter(name),
            Named(ref name) => ClassQuery::Binary(name),
            NamedValue { ref name, ref value, .. } => ClassQuery::ByValue {
                property_name: name,
                property_value: value,
            },
        };
        let mut result = self.convert_unicode_class_error(
            &ast_class.span,
            unicode::class(query),
        );
        if let Ok(ref mut class) = result {
            self.unicode_fold_and_negate(
                &ast_class.span,
                ast_class.negated,
                class,
            )?;
        }
        result
    }

    fn hir_ascii_unicode_class(
        &self,
        ast: &ast::ClassAscii,
    ) -> Result<hir::ClassUnicode> {
        let mut cls = hir::ClassUnicode::new(
            ascii_class_as_chars(&ast.kind)
                .map(|(s, e)| hir::ClassUnicodeRange::new(s, e)),
        );
        self.unicode_fold_and_negate(&ast.span, ast.negated, &mut cls)?;
        Ok(cls)
    }

    fn hir_ascii_byte_class(
        &self,
        ast: &ast::ClassAscii,
    ) -> Result<hir::ClassBytes> {
        let mut cls = hir::ClassBytes::new(
            ascii_class(&ast.kind)
                .map(|(s, e)| hir::ClassBytesRange::new(s, e)),
        );
        self.bytes_fold_and_negate(&ast.span, ast.negated, &mut cls)?;
        Ok(cls)
    }

    fn hir_perl_unicode_class(
        &self,
        ast_class: &ast::ClassPerl,
    ) -> Result<hir::ClassUnicode> {
        use crate::ast::ClassPerlKind::*;

        assert!(self.flags().unicode());
        let result = match ast_class.kind {
            Digit => unicode::perl_digit(),
            Space => unicode::perl_space(),
            Word => unicode::perl_word(),
        };
        let mut class =
            self.convert_unicode_class_error(&ast_class.span, result)?;
        // We needn't apply case folding here because the Perl Unicode classes
        // are already closed under Unicode simple case folding.
        if ast_class.negated {
            class.negate();
        }
        Ok(class)
    }

    fn hir_perl_byte_class(
        &self,
        ast_class: &ast::ClassPerl,
    ) -> Result<hir::ClassBytes> {
        use crate::ast::ClassPerlKind::*;

        assert!(!self.flags().unicode());
        let mut class = match ast_class.kind {
            Digit => hir_ascii_class_bytes(&ast::ClassAsciiKind::Digit),
            Space => hir_ascii_class_bytes(&ast::ClassAsciiKind::Space),
            Word => hir_ascii_class_bytes(&ast::ClassAsciiKind::Word),
        };
        // We needn't apply case folding here because the Perl ASCII classes
        // are already closed (under ASCII case folding).
        if ast_class.negated {
            class.negate();
        }
        // Negating a Perl byte class is likely to cause it to match invalid
        // UTF-8. That's only OK if the translator is configured to allow such
        // things.
        if self.trans().utf8 && !class.is_ascii() {
            return Err(self.error(ast_class.span, ErrorKind::InvalidUtf8));
        }
        Ok(class)
    }

    /// Converts the given Unicode specific error to an HIR translation error.
    ///
    /// The span given should approximate the position at which an error would
    /// occur.
    fn convert_unicode_class_error(
        &self,
        span: &Span,
        result: core::result::Result<hir::ClassUnicode, unicode::Error>,
    ) -> Result<hir::ClassUnicode> {
        result.map_err(|err| {
            let sp = span.clone();
            match err {
                unicode::Error::PropertyNotFound => {
                    self.error(sp, ErrorKind::UnicodePropertyNotFound)
                }
                unicode::Error::PropertyValueNotFound => {
                    self.error(sp, ErrorKind::UnicodePropertyValueNotFound)
                }
                unicode::Error::PerlClassNotFound => {
                    self.error(sp, ErrorKind::UnicodePerlClassNotFound)
                }
            }
        })
    }

    fn unicode_fold_and_negate(
        &self,
        span: &Span,
        negated: bool,
        class: &mut hir::ClassUnicode,
    ) -> Result<()> {
        // Note that we must apply case folding before negation!
        // Consider `(?i)[^x]`. If we applied negation first, then
        // the result would be the character class that matched any
        // Unicode scalar value.
        if self.flags().case_insensitive() {
            class.try_case_fold_simple().map_err(|_| {
                self.error(span.clone(), ErrorKind::UnicodeCaseUnavailable)
            })?;
        }
        if negated {
            class.negate();
        }
        Ok(())
    }

    fn bytes_fold_and_negate(
        &self,
        span: &Span,
        negated: bool,
        class: &mut hir::ClassBytes,
    ) -> Result<()> {
        // Note that we must apply case folding before negation!
        // Consider `(?i)[^x]`. If we applied negation first, then
        // the result would be the character class that matched any
        // Unicode scalar value.
        if self.flags().case_insensitive() {
            class.case_fold_simple();
        }
        if negated {
            class.negate();
        }
        if self.trans().utf8 && !class.is_ascii() {
            return Err(self.error(span.clone(), ErrorKind::InvalidUtf8));
        }
        Ok(())
    }

    /// Return a scalar byte value suitable for use as a literal in a byte
    /// character class.
    fn class_literal_byte(&self, ast: &ast::Literal) -> Result<u8> {
        match self.ast_literal_to_scalar(ast)? {
            Either::Right(byte) => Ok(byte),
            Either::Left(ch) => {
                if ch.is_ascii() {
                    Ok(u8::try_from(ch).unwrap())
                } else {
                    // We can't feasibly support Unicode in
                    // byte oriented classes. Byte classes don't
                    // do Unicode case folding.
                    Err(self.error(ast.span, ErrorKind::UnicodeNotAllowed))
                }
            }
        }
    }
}

/// A translator's representation of a regular expression's flags at any given
/// moment in time.
///
/// Each flag can be in one of three states: absent, present but disabled or
/// present but enabled.
#[derive(Clone, Copy, Debug, Default)]
struct Flags {
    case_insensitive: Option<bool>,
    multi_line: Option<bool>,
    dot_matches_new_line: Option<bool>,
    swap_greed: Option<bool>,
    unicode: Option<bool>,
    crlf: Option<bool>,
    // Note that `ignore_whitespace` is omitted here because it is handled
    // entirely in the parser.
}

/// Represents a transformation on Flags, used by `compute_exit_flags`.
///
/// This type allows expressing flag transformations as data, which can be
/// composed and then applied. This enables a catamorphism-based implementation
/// of `compute_exit_flags` that is stack-safe.
#[derive(Clone, Debug)]
enum FlagOp<'a> {
    /// No change to flags (identity transformation).
    Identity,
    /// Set flags from an AST Flags node (merges with input).
    SetFlags(&'a ast::Flags),
    /// Compose multiple operations left-to-right.
    Compose(Vec<FlagOp<'a>>),
}

impl FlagOp<'_> {
    /// Apply this flag operation to the given flags.
    fn apply(self, flags: Flags) -> Flags {
        match self {
            FlagOp::Identity => flags,
            FlagOp::SetFlags(ast_flags) => {
                let mut new_flags = Flags::from_ast(ast_flags);
                new_flags.merge(&flags);
                new_flags
            }
            FlagOp::Compose(ops) => {
                ops.into_iter().fold(flags, |f, op| op.apply(f))
            }
        }
    }
}

impl Flags {
    fn from_ast(ast: &ast::Flags) -> Flags {
        let mut flags = Flags::default();
        let mut enable = true;
        for item in &ast.items {
            match item.kind {
                ast::FlagsItemKind::Negation => {
                    enable = false;
                }
                ast::FlagsItemKind::Flag(ast::Flag::CaseInsensitive) => {
                    flags.case_insensitive = Some(enable);
                }
                ast::FlagsItemKind::Flag(ast::Flag::MultiLine) => {
                    flags.multi_line = Some(enable);
                }
                ast::FlagsItemKind::Flag(ast::Flag::DotMatchesNewLine) => {
                    flags.dot_matches_new_line = Some(enable);
                }
                ast::FlagsItemKind::Flag(ast::Flag::SwapGreed) => {
                    flags.swap_greed = Some(enable);
                }
                ast::FlagsItemKind::Flag(ast::Flag::Unicode) => {
                    flags.unicode = Some(enable);
                }
                ast::FlagsItemKind::Flag(ast::Flag::CRLF) => {
                    flags.crlf = Some(enable);
                }
                ast::FlagsItemKind::Flag(ast::Flag::IgnoreWhitespace) => {}
            }
        }
        flags
    }

    fn merge(&mut self, previous: &Flags) {
        if self.case_insensitive.is_none() {
            self.case_insensitive = previous.case_insensitive;
        }
        if self.multi_line.is_none() {
            self.multi_line = previous.multi_line;
        }
        if self.dot_matches_new_line.is_none() {
            self.dot_matches_new_line = previous.dot_matches_new_line;
        }
        if self.swap_greed.is_none() {
            self.swap_greed = previous.swap_greed;
        }
        if self.unicode.is_none() {
            self.unicode = previous.unicode;
        }
        if self.crlf.is_none() {
            self.crlf = previous.crlf;
        }
    }

    fn case_insensitive(&self) -> bool {
        self.case_insensitive.unwrap_or(false)
    }

    fn multi_line(&self) -> bool {
        self.multi_line.unwrap_or(false)
    }

    fn dot_matches_new_line(&self) -> bool {
        self.dot_matches_new_line.unwrap_or(false)
    }

    fn swap_greed(&self) -> bool {
        self.swap_greed.unwrap_or(false)
    }

    fn unicode(&self) -> bool {
        self.unicode.unwrap_or(true)
    }

    fn crlf(&self) -> bool {
        self.crlf.unwrap_or(false)
    }
}

fn hir_ascii_class_bytes(kind: &ast::ClassAsciiKind) -> hir::ClassBytes {
    let ranges: Vec<_> = ascii_class(kind)
        .map(|(s, e)| hir::ClassBytesRange::new(s, e))
        .collect();
    hir::ClassBytes::new(ranges)
}

fn ascii_class(kind: &ast::ClassAsciiKind) -> impl Iterator<Item = (u8, u8)> {
    use crate::ast::ClassAsciiKind::*;

    let slice: &'static [(u8, u8)] = match *kind {
        Alnum => &[(b'0', b'9'), (b'A', b'Z'), (b'a', b'z')],
        Alpha => &[(b'A', b'Z'), (b'a', b'z')],
        Ascii => &[(b'\x00', b'\x7F')],
        Blank => &[(b'\t', b'\t'), (b' ', b' ')],
        Cntrl => &[(b'\x00', b'\x1F'), (b'\x7F', b'\x7F')],
        Digit => &[(b'0', b'9')],
        Graph => &[(b'!', b'~')],
        Lower => &[(b'a', b'z')],
        Print => &[(b' ', b'~')],
        Punct => &[(b'!', b'/'), (b':', b'@'), (b'[', b'`'), (b'{', b'~')],
        Space => &[
            (b'\t', b'\t'),
            (b'\n', b'\n'),
            (b'\x0B', b'\x0B'),
            (b'\x0C', b'\x0C'),
            (b'\r', b'\r'),
            (b' ', b' '),
        ],
        Upper => &[(b'A', b'Z')],
        Word => &[(b'0', b'9'), (b'A', b'Z'), (b'_', b'_'), (b'a', b'z')],
        Xdigit => &[(b'0', b'9'), (b'A', b'F'), (b'a', b'f')],
    };
    slice.iter().copied()
}

fn ascii_class_as_chars(
    kind: &ast::ClassAsciiKind,
) -> impl Iterator<Item = (char, char)> {
    ascii_class(kind).map(|(s, e)| (char::from(s), char::from(e)))
}

#[cfg(test)]
mod tests {
    use crate::{
        ast::{parse::ParserBuilder, Position},
        hir::{Look, Properties},
    };

    use super::*;

    // We create these errors to compare with real hir::Errors in the tests.
    // We define equality between TestError and hir::Error to disregard the
    // pattern string in hir::Error, which is annoying to provide in tests.
    #[derive(Clone, Debug)]
    struct TestError {
        span: Span,
        kind: hir::ErrorKind,
    }

    impl PartialEq<hir::Error> for TestError {
        fn eq(&self, other: &hir::Error) -> bool {
            self.span == other.span && self.kind == other.kind
        }
    }

    impl PartialEq<TestError> for hir::Error {
        fn eq(&self, other: &TestError) -> bool {
            self.span == other.span && self.kind == other.kind
        }
    }

    fn parse(pattern: &str) -> Ast {
        ParserBuilder::new().octal(true).build().parse(pattern).unwrap()
    }

    fn t(pattern: &str) -> Hir {
        TranslatorBuilder::new()
            .utf8(true)
            .build()
            .translate(pattern, &parse(pattern))
            .unwrap()
    }

    fn t_err(pattern: &str) -> hir::Error {
        TranslatorBuilder::new()
            .utf8(true)
            .build()
            .translate(pattern, &parse(pattern))
            .unwrap_err()
    }

    fn t_bytes(pattern: &str) -> Hir {
        TranslatorBuilder::new()
            .utf8(false)
            .build()
            .translate(pattern, &parse(pattern))
            .unwrap()
    }

    fn props(pattern: &str) -> Properties {
        t(pattern).properties().clone()
    }

    fn props_bytes(pattern: &str) -> Properties {
        t_bytes(pattern).properties().clone()
    }

    fn hir_lit(s: &str) -> Hir {
        hir_blit(s.as_bytes())
    }

    fn hir_blit(s: &[u8]) -> Hir {
        Hir::literal(s)
    }

    fn hir_capture(index: u32, expr: Hir) -> Hir {
        Hir::capture(hir::Capture { index, name: None, sub: Box::new(expr) })
    }

    fn hir_capture_name(index: u32, name: &str, expr: Hir) -> Hir {
        Hir::capture(hir::Capture {
            index,
            name: Some(name.into()),
            sub: Box::new(expr),
        })
    }

    fn hir_quest(greedy: bool, expr: Hir) -> Hir {
        Hir::repetition(hir::Repetition {
            min: 0,
            max: Some(1),
            greedy,
            sub: Box::new(expr),
        })
    }

    fn hir_star(greedy: bool, expr: Hir) -> Hir {
        Hir::repetition(hir::Repetition {
            min: 0,
            max: None,
            greedy,
            sub: Box::new(expr),
        })
    }

    fn hir_plus(greedy: bool, expr: Hir) -> Hir {
        Hir::repetition(hir::Repetition {
            min: 1,
            max: None,
            greedy,
            sub: Box::new(expr),
        })
    }

    fn hir_range(greedy: bool, min: u32, max: Option<u32>, expr: Hir) -> Hir {
        Hir::repetition(hir::Repetition {
            min,
            max,
            greedy,
            sub: Box::new(expr),
        })
    }

    fn hir_alt(alts: Vec<Hir>) -> Hir {
        Hir::alternation(alts)
    }

    fn hir_cat(exprs: Vec<Hir>) -> Hir {
        Hir::concat(exprs)
    }

    #[allow(dead_code)]
    fn hir_uclass_query(query: ClassQuery<'_>) -> Hir {
        Hir::class(hir::Class::Unicode(unicode::class(query).unwrap()))
    }

    #[allow(dead_code)]
    fn hir_uclass_perl_word() -> Hir {
        Hir::class(hir::Class::Unicode(unicode::perl_word().unwrap()))
    }

    fn hir_ascii_uclass(kind: &ast::ClassAsciiKind) -> Hir {
        Hir::class(hir::Class::Unicode(hir::ClassUnicode::new(
            ascii_class_as_chars(kind)
                .map(|(s, e)| hir::ClassUnicodeRange::new(s, e)),
        )))
    }

    fn hir_ascii_bclass(kind: &ast::ClassAsciiKind) -> Hir {
        Hir::class(hir::Class::Bytes(hir::ClassBytes::new(
            ascii_class(kind).map(|(s, e)| hir::ClassBytesRange::new(s, e)),
        )))
    }

    fn hir_uclass(ranges: &[(char, char)]) -> Hir {
        Hir::class(uclass(ranges))
    }

    fn hir_bclass(ranges: &[(u8, u8)]) -> Hir {
        Hir::class(bclass(ranges))
    }

    fn hir_case_fold(expr: Hir) -> Hir {
        match expr.into_kind() {
            HirKind::Class(mut cls) => {
                cls.case_fold_simple();
                Hir::class(cls)
            }
            _ => panic!("cannot case fold non-class Hir expr"),
        }
    }

    fn hir_negate(expr: Hir) -> Hir {
        match expr.into_kind() {
            HirKind::Class(mut cls) => {
                cls.negate();
                Hir::class(cls)
            }
            _ => panic!("cannot negate non-class Hir expr"),
        }
    }

    fn uclass(ranges: &[(char, char)]) -> hir::Class {
        let ranges: Vec<hir::ClassUnicodeRange> = ranges
            .iter()
            .map(|&(s, e)| hir::ClassUnicodeRange::new(s, e))
            .collect();
        hir::Class::Unicode(hir::ClassUnicode::new(ranges))
    }

    fn bclass(ranges: &[(u8, u8)]) -> hir::Class {
        let ranges: Vec<hir::ClassBytesRange> = ranges
            .iter()
            .map(|&(s, e)| hir::ClassBytesRange::new(s, e))
            .collect();
        hir::Class::Bytes(hir::ClassBytes::new(ranges))
    }

    #[cfg(feature = "unicode-case")]
    fn class_case_fold(mut cls: hir::Class) -> Hir {
        cls.case_fold_simple();
        Hir::class(cls)
    }

    fn class_negate(mut cls: hir::Class) -> Hir {
        cls.negate();
        Hir::class(cls)
    }

    #[allow(dead_code)]
    fn hir_union(expr1: Hir, expr2: Hir) -> Hir {
        use crate::hir::Class::{Bytes, Unicode};

        match (expr1.into_kind(), expr2.into_kind()) {
            (HirKind::Class(Unicode(mut c1)), HirKind::Class(Unicode(c2))) => {
                c1.union(&c2);
                Hir::class(hir::Class::Unicode(c1))
            }
            (HirKind::Class(Bytes(mut c1)), HirKind::Class(Bytes(c2))) => {
                c1.union(&c2);
                Hir::class(hir::Class::Bytes(c1))
            }
            _ => panic!("cannot union non-class Hir exprs"),
        }
    }

    #[allow(dead_code)]
    fn hir_difference(expr1: Hir, expr2: Hir) -> Hir {
        use crate::hir::Class::{Bytes, Unicode};

        match (expr1.into_kind(), expr2.into_kind()) {
            (HirKind::Class(Unicode(mut c1)), HirKind::Class(Unicode(c2))) => {
                c1.difference(&c2);
                Hir::class(hir::Class::Unicode(c1))
            }
            (HirKind::Class(Bytes(mut c1)), HirKind::Class(Bytes(c2))) => {
                c1.difference(&c2);
                Hir::class(hir::Class::Bytes(c1))
            }
            _ => panic!("cannot difference non-class Hir exprs"),
        }
    }

    fn hir_look(look: hir::Look) -> Hir {
        Hir::look(look)
    }

    #[test]
    fn empty() {
        assert_eq!(t(""), Hir::empty());
        assert_eq!(t("(?i)"), Hir::empty());
        assert_eq!(t("()"), hir_capture(1, Hir::empty()));
        assert_eq!(t("(?:)"), Hir::empty());
        assert_eq!(t("(?P<wat>)"), hir_capture_name(1, "wat", Hir::empty()));
        assert_eq!(t("|"), hir_alt(vec![Hir::empty(), Hir::empty()]));
        assert_eq!(
            t("()|()"),
            hir_alt(vec![
                hir_capture(1, Hir::empty()),
                hir_capture(2, Hir::empty()),
            ])
        );
        assert_eq!(
            t("(|b)"),
            hir_capture(1, hir_alt(vec![Hir::empty(), hir_lit("b"),]))
        );
        assert_eq!(
            t("(a|)"),
            hir_capture(1, hir_alt(vec![hir_lit("a"), Hir::empty(),]))
        );
        assert_eq!(
            t("(a||c)"),
            hir_capture(
                1,
                hir_alt(vec![hir_lit("a"), Hir::empty(), hir_lit("c"),])
            )
        );
        assert_eq!(
            t("(||)"),
            hir_capture(
                1,
                hir_alt(vec![Hir::empty(), Hir::empty(), Hir::empty(),])
            )
        );
    }

    #[test]
    fn literal() {
        assert_eq!(t("a"), hir_lit("a"));
        assert_eq!(t("(?-u)a"), hir_lit("a"));
        assert_eq!(t("☃"), hir_lit("☃"));
        assert_eq!(t("abcd"), hir_lit("abcd"));

        assert_eq!(t_bytes("(?-u)a"), hir_lit("a"));
        assert_eq!(t_bytes("(?-u)\x61"), hir_lit("a"));
        assert_eq!(t_bytes(r"(?-u)\x61"), hir_lit("a"));
        assert_eq!(t_bytes(r"(?-u)\xFF"), hir_blit(b"\xFF"));

        assert_eq!(t("(?-u)☃"), hir_lit("☃"));
        assert_eq!(
            t_err(r"(?-u)\xFF"),
            TestError {
                kind: hir::ErrorKind::InvalidUtf8,
                span: Span::new(
                    Position::new(5, 1, 6),
                    Position::new(9, 1, 10)
                ),
            }
        );
    }

    #[test]
    fn literal_case_insensitive() {
        #[cfg(feature = "unicode-case")]
        assert_eq!(t("(?i)a"), hir_uclass(&[('A', 'A'), ('a', 'a'),]));
        #[cfg(feature = "unicode-case")]
        assert_eq!(t("(?i:a)"), hir_uclass(&[('A', 'A'), ('a', 'a')]));
        #[cfg(feature = "unicode-case")]
        assert_eq!(
            t("a(?i)a(?-i)a"),
            hir_cat(vec![
                hir_lit("a"),
                hir_uclass(&[('A', 'A'), ('a', 'a')]),
                hir_lit("a"),
            ])
        );
        #[cfg(feature = "unicode-case")]
        assert_eq!(
            t("(?i)ab@c"),
            hir_cat(vec![
                hir_uclass(&[('A', 'A'), ('a', 'a')]),
                hir_uclass(&[('B', 'B'), ('b', 'b')]),
                hir_lit("@"),
                hir_uclass(&[('C', 'C'), ('c', 'c')]),
            ])
        );
        #[cfg(feature = "unicode-case")]
        assert_eq!(
            t("(?i)β"),
            hir_uclass(&[('Β', 'Β'), ('β', 'β'), ('ϐ', 'ϐ'),])
        );

        assert_eq!(t("(?i-u)a"), hir_bclass(&[(b'A', b'A'), (b'a', b'a'),]));
        #[cfg(feature = "unicode-case")]
        assert_eq!(
            t("(?-u)a(?i)a(?-i)a"),
            hir_cat(vec![
                hir_lit("a"),
                hir_bclass(&[(b'A', b'A'), (b'a', b'a')]),
                hir_lit("a"),
            ])
        );
        assert_eq!(
            t("(?i-u)ab@c"),
            hir_cat(vec![
                hir_bclass(&[(b'A', b'A'), (b'a', b'a')]),
                hir_bclass(&[(b'B', b'B'), (b'b', b'b')]),
                hir_lit("@"),
                hir_bclass(&[(b'C', b'C'), (b'c', b'c')]),
            ])
        );

        assert_eq!(
            t_bytes("(?i-u)a"),
            hir_bclass(&[(b'A', b'A'), (b'a', b'a'),])
        );
        assert_eq!(
            t_bytes("(?i-u)\x61"),
            hir_bclass(&[(b'A', b'A'), (b'a', b'a'),])
        );
        assert_eq!(
            t_bytes(r"(?i-u)\x61"),
            hir_bclass(&[(b'A', b'A'), (b'a', b'a'),])
        );
        assert_eq!(t_bytes(r"(?i-u)\xFF"), hir_blit(b"\xFF"));

        assert_eq!(t("(?i-u)β"), hir_lit("β"),);
    }

    #[test]
    fn dot() {
        assert_eq!(
            t("."),
            hir_uclass(&[('\0', '\t'), ('\x0B', '\u{10FFFF}')])
        );
        assert_eq!(
            t("(?R)."),
            hir_uclass(&[
                ('\0', '\t'),
                ('\x0B', '\x0C'),
                ('\x0E', '\u{10FFFF}'),
            ])
        );
        assert_eq!(t("(?s)."), hir_uclass(&[('\0', '\u{10FFFF}')]));
        assert_eq!(t("(?Rs)."), hir_uclass(&[('\0', '\u{10FFFF}')]));
        assert_eq!(
            t_bytes("(?-u)."),
            hir_bclass(&[(b'\0', b'\t'), (b'\x0B', b'\xFF')])
        );
        assert_eq!(
            t_bytes("(?R-u)."),
            hir_bclass(&[
                (b'\0', b'\t'),
                (b'\x0B', b'\x0C'),
                (b'\x0E', b'\xFF'),
            ])
        );
        assert_eq!(t_bytes("(?s-u)."), hir_bclass(&[(b'\0', b'\xFF'),]));
        assert_eq!(t_bytes("(?Rs-u)."), hir_bclass(&[(b'\0', b'\xFF'),]));

        // If invalid UTF-8 isn't allowed, then non-Unicode `.` isn't allowed.
        assert_eq!(
            t_err("(?-u)."),
            TestError {
                kind: hir::ErrorKind::InvalidUtf8,
                span: Span::new(
                    Position::new(5, 1, 6),
                    Position::new(6, 1, 7)
                ),
            }
        );
        assert_eq!(
            t_err("(?R-u)."),
            TestError {
                kind: hir::ErrorKind::InvalidUtf8,
                span: Span::new(
                    Position::new(6, 1, 7),
                    Position::new(7, 1, 8)
                ),
            }
        );
        assert_eq!(
            t_err("(?s-u)."),
            TestError {
                kind: hir::ErrorKind::InvalidUtf8,
                span: Span::new(
                    Position::new(6, 1, 7),
                    Position::new(7, 1, 8)
                ),
            }
        );
        assert_eq!(
            t_err("(?Rs-u)."),
            TestError {
                kind: hir::ErrorKind::InvalidUtf8,
                span: Span::new(
                    Position::new(7, 1, 8),
                    Position::new(8, 1, 9)
                ),
            }
        );
    }

    #[test]
    fn assertions() {
        assert_eq!(t("^"), hir_look(hir::Look::Start));
        assert_eq!(t("$"), hir_look(hir::Look::End));
        assert_eq!(t(r"\A"), hir_look(hir::Look::Start));
        assert_eq!(t(r"\z"), hir_look(hir::Look::End));
        assert_eq!(t("(?m)^"), hir_look(hir::Look::StartLF));
        assert_eq!(t("(?m)$"), hir_look(hir::Look::EndLF));
        assert_eq!(t(r"(?m)\A"), hir_look(hir::Look::Start));
        assert_eq!(t(r"(?m)\z"), hir_look(hir::Look::End));

        assert_eq!(t(r"\b"), hir_look(hir::Look::WordUnicode));
        assert_eq!(t(r"\B"), hir_look(hir::Look::WordUnicodeNegate));
        assert_eq!(t(r"(?-u)\b"), hir_look(hir::Look::WordAscii));
        assert_eq!(t(r"(?-u)\B"), hir_look(hir::Look::WordAsciiNegate));
    }

    #[test]
    fn group() {
        assert_eq!(t("(a)"), hir_capture(1, hir_lit("a")));
        assert_eq!(
            t("(a)(b)"),
            hir_cat(vec![
                hir_capture(1, hir_lit("a")),
                hir_capture(2, hir_lit("b")),
            ])
        );
        assert_eq!(
            t("(a)|(b)"),
            hir_alt(vec![
                hir_capture(1, hir_lit("a")),
                hir_capture(2, hir_lit("b")),
            ])
        );
        assert_eq!(t("(?P<foo>)"), hir_capture_name(1, "foo", Hir::empty()));
        assert_eq!(t("(?P<foo>a)"), hir_capture_name(1, "foo", hir_lit("a")));
        assert_eq!(
            t("(?P<foo>a)(?P<bar>b)"),
            hir_cat(vec![
                hir_capture_name(1, "foo", hir_lit("a")),
                hir_capture_name(2, "bar", hir_lit("b")),
            ])
        );
        assert_eq!(t("(?:)"), Hir::empty());
        assert_eq!(t("(?:a)"), hir_lit("a"));
        assert_eq!(
            t("(?:a)(b)"),
            hir_cat(vec![hir_lit("a"), hir_capture(1, hir_lit("b")),])
        );
        assert_eq!(
            t("(a)(?:b)(c)"),
            hir_cat(vec![
                hir_capture(1, hir_lit("a")),
                hir_lit("b"),
                hir_capture(2, hir_lit("c")),
            ])
        );
        assert_eq!(
            t("(a)(?P<foo>b)(c)"),
            hir_cat(vec![
                hir_capture(1, hir_lit("a")),
                hir_capture_name(2, "foo", hir_lit("b")),
                hir_capture(3, hir_lit("c")),
            ])
        );
        assert_eq!(t("()"), hir_capture(1, Hir::empty()));
        assert_eq!(t("((?i))"), hir_capture(1, Hir::empty()));
        assert_eq!(t("((?x))"), hir_capture(1, Hir::empty()));
        assert_eq!(
            t("(((?x)))"),
            hir_capture(1, hir_capture(2, Hir::empty()))
        );
    }

    #[test]
    fn line_anchors() {
        assert_eq!(t("^"), hir_look(hir::Look::Start));
        assert_eq!(t("$"), hir_look(hir::Look::End));
        assert_eq!(t(r"\A"), hir_look(hir::Look::Start));
        assert_eq!(t(r"\z"), hir_look(hir::Look::End));

        assert_eq!(t(r"(?m)\A"), hir_look(hir::Look::Start));
        assert_eq!(t(r"(?m)\z"), hir_look(hir::Look::End));
        assert_eq!(t("(?m)^"), hir_look(hir::Look::StartLF));
        assert_eq!(t("(?m)$"), hir_look(hir::Look::EndLF));

        assert_eq!(t(r"(?R)\A"), hir_look(hir::Look::Start));
        assert_eq!(t(r"(?R)\z"), hir_look(hir::Look::End));
        assert_eq!(t("(?R)^"), hir_look(hir::Look::Start));
        assert_eq!(t("(?R)$"), hir_look(hir::Look::End));

        assert_eq!(t(r"(?Rm)\A"), hir_look(hir::Look::Start));
        assert_eq!(t(r"(?Rm)\z"), hir_look(hir::Look::End));
        assert_eq!(t("(?Rm)^"), hir_look(hir::Look::StartCRLF));
        assert_eq!(t("(?Rm)$"), hir_look(hir::Look::EndCRLF));
    }

    #[test]
    fn flags() {
        #[cfg(feature = "unicode-case")]
        assert_eq!(
            t("(?i:a)a"),
            hir_cat(
                vec![hir_uclass(&[('A', 'A'), ('a', 'a')]), hir_lit("a"),]
            )
        );
        assert_eq!(
            t("(?i-u:a)β"),
            hir_cat(vec![
                hir_bclass(&[(b'A', b'A'), (b'a', b'a')]),
                hir_lit("β"),
            ])
        );
        assert_eq!(
            t("(?:(?i-u)a)b"),
            hir_cat(vec![
                hir_bclass(&[(b'A', b'A'), (b'a', b'a')]),
                hir_lit("b"),
            ])
        );
        assert_eq!(
            t("((?i-u)a)b"),
            hir_cat(vec![
                hir_capture(1, hir_bclass(&[(b'A', b'A'), (b'a', b'a')])),
                hir_lit("b"),
            ])
        );
        #[cfg(feature = "unicode-case")]
        assert_eq!(
            t("(?i)(?-i:a)a"),
            hir_cat(
                vec![hir_lit("a"), hir_uclass(&[('A', 'A'), ('a', 'a')]),]
            )
        );
        #[cfg(feature = "unicode-case")]
        assert_eq!(
            t("(?im)a^"),
            hir_cat(vec![
                hir_uclass(&[('A', 'A'), ('a', 'a')]),
                hir_look(hir::Look::StartLF),
            ])
        );
        #[cfg(feature = "unicode-case")]
        assert_eq!(
            t("(?im)a^(?i-m)a^"),
            hir_cat(vec![
                hir_uclass(&[('A', 'A'), ('a', 'a')]),
                hir_look(hir::Look::StartLF),
                hir_uclass(&[('A', 'A'), ('a', 'a')]),
                hir_look(hir::Look::Start),
            ])
        );
        assert_eq!(
            t("(?U)a*a*?(?-U)a*a*?"),
            hir_cat(vec![
                hir_star(false, hir_lit("a")),
                hir_star(true, hir_lit("a")),
                hir_star(true, hir_lit("a")),
                hir_star(false, hir_lit("a")),
            ])
        );
        #[cfg(feature = "unicode-case")]
        assert_eq!(
            t("(?:a(?i)a)a"),
            hir_cat(vec![
                hir_cat(vec![
                    hir_lit("a"),
                    hir_uclass(&[('A', 'A'), ('a', 'a')]),
                ]),
                hir_lit("a"),
            ])
        );
        #[cfg(feature = "unicode-case")]
        assert_eq!(
            t("(?i)(?:a(?-i)a)a"),
            hir_cat(vec![
                hir_cat(vec![
                    hir_uclass(&[('A', 'A'), ('a', 'a')]),
                    hir_lit("a"),
                ]),
                hir_uclass(&[('A', 'A'), ('a', 'a')]),
            ])
        );
    }

    #[test]
    fn escape() {
        assert_eq!(
            t(r"\\\.\+\*\?\(\)\|\[\]\{\}\^\$\#"),
            hir_lit(r"\.+*?()|[]{}^$#")
        );
    }

    #[test]
    fn repetition() {
        assert_eq!(t("a?"), hir_quest(true, hir_lit("a")));
        assert_eq!(t("a*"), hir_star(true, hir_lit("a")));
        assert_eq!(t("a+"), hir_plus(true, hir_lit("a")));
        assert_eq!(t("a??"), hir_quest(false, hir_lit("a")));
        assert_eq!(t("a*?"), hir_star(false, hir_lit("a")));
        assert_eq!(t("a+?"), hir_plus(false, hir_lit("a")));

        assert_eq!(t("a{1}"), hir_range(true, 1, Some(1), hir_lit("a"),));
        assert_eq!(t("a{1,}"), hir_range(true, 1, None, hir_lit("a"),));
        assert_eq!(t("a{1,2}"), hir_range(true, 1, Some(2), hir_lit("a"),));
        assert_eq!(t("a{1}?"), hir_range(false, 1, Some(1), hir_lit("a"),));
        assert_eq!(t("a{1,}?"), hir_range(false, 1, None, hir_lit("a"),));
        assert_eq!(t("a{1,2}?"), hir_range(false, 1, Some(2), hir_lit("a"),));

        assert_eq!(
            t("ab?"),
            hir_cat(vec![hir_lit("a"), hir_quest(true, hir_lit("b")),])
        );
        assert_eq!(t("(ab)?"), hir_quest(true, hir_capture(1, hir_lit("ab"))));
        assert_eq!(
            t("a|b?"),
            hir_alt(vec![hir_lit("a"), hir_quest(true, hir_lit("b")),])
        );
    }

    #[test]
    fn cat_alt() {
        let a = || hir_look(hir::Look::Start);
        let b = || hir_look(hir::Look::End);
        let c = || hir_look(hir::Look::WordUnicode);
        let d = || hir_look(hir::Look::WordUnicodeNegate);

        assert_eq!(t("(^$)"), hir_capture(1, hir_cat(vec![a(), b()])));
        assert_eq!(t("^|$"), hir_alt(vec![a(), b()]));
        assert_eq!(t(r"^|$|\b"), hir_alt(vec![a(), b(), c()]));
        assert_eq!(
            t(r"^$|$\b|\b\B"),
            hir_alt(vec![
                hir_cat(vec![a(), b()]),
                hir_cat(vec![b(), c()]),
                hir_cat(vec![c(), d()]),
            ])
        );
        assert_eq!(t("(^|$)"), hir_capture(1, hir_alt(vec![a(), b()])));
        assert_eq!(
            t(r"(^|$|\b)"),
            hir_capture(1, hir_alt(vec![a(), b(), c()]))
        );
        assert_eq!(
            t(r"(^$|$\b|\b\B)"),
            hir_capture(
                1,
                hir_alt(vec![
                    hir_cat(vec![a(), b()]),
                    hir_cat(vec![b(), c()]),
                    hir_cat(vec![c(), d()]),
                ])
            )
        );
        assert_eq!(
            t(r"(^$|($\b|(\b\B)))"),
            hir_capture(
                1,
                hir_alt(vec![
                    hir_cat(vec![a(), b()]),
                    hir_capture(
                        2,
                        hir_alt(vec![
                            hir_cat(vec![b(), c()]),
                            hir_capture(3, hir_cat(vec![c(), d()])),
                        ])
                    ),
                ])
            )
        );
    }

    // Tests the HIR transformation of things like '[a-z]|[A-Z]' into
    // '[A-Za-z]'. In other words, an alternation of just classes is always
    // equivalent to a single class corresponding to the union of the branches
    // in that class. (Unless some branches match invalid UTF-8 and others
    // match non-ASCII Unicode.)
    #[test]
    fn cat_class_flattened() {
        assert_eq!(t(r"[a-z]|[A-Z]"), hir_uclass(&[('A', 'Z'), ('a', 'z')]));
        // Combining all of the letter properties should give us the one giant
        // letter property.
        #[cfg(feature = "unicode-gencat")]
        assert_eq!(
            t(r"(?x)
                \p{Lowercase_Letter}
                |\p{Uppercase_Letter}
                |\p{Titlecase_Letter}
                |\p{Modifier_Letter}
                |\p{Other_Letter}
            "),
            hir_uclass_query(ClassQuery::Binary("letter"))
        );
        // Byte classes that can truly match invalid UTF-8 cannot be combined
        // with Unicode classes.
        assert_eq!(
            t_bytes(r"[Δδ]|(?-u:[\x90-\xFF])|[Λλ]"),
            hir_alt(vec![
                hir_uclass(&[('Δ', 'Δ'), ('δ', 'δ')]),
                hir_bclass(&[(b'\x90', b'\xFF')]),
                hir_uclass(&[('Λ', 'Λ'), ('λ', 'λ')]),
            ])
        );
        // Byte classes on their own can be combined, even if some are ASCII
        // and others are invalid UTF-8.
        assert_eq!(
            t_bytes(r"[a-z]|(?-u:[\x90-\xFF])|[A-Z]"),
            hir_bclass(&[(b'A', b'Z'), (b'a', b'z'), (b'\x90', b'\xFF')]),
        );
    }

    #[test]
    fn class_ascii() {
        assert_eq!(
            t("[[:alnum:]]"),
            hir_ascii_uclass(&ast::ClassAsciiKind::Alnum)
        );
        assert_eq!(
            t("[[:alpha:]]"),
            hir_ascii_uclass(&ast::ClassAsciiKind::Alpha)
        );
        assert_eq!(
            t("[[:ascii:]]"),
            hir_ascii_uclass(&ast::ClassAsciiKind::Ascii)
        );
        assert_eq!(
            t("[[:blank:]]"),
            hir_ascii_uclass(&ast::ClassAsciiKind::Blank)
        );
        assert_eq!(
            t("[[:cntrl:]]"),
            hir_ascii_uclass(&ast::ClassAsciiKind::Cntrl)
        );
        assert_eq!(
            t("[[:digit:]]"),
            hir_ascii_uclass(&ast::ClassAsciiKind::Digit)
        );
        assert_eq!(
            t("[[:graph:]]"),
            hir_ascii_uclass(&ast::ClassAsciiKind::Graph)
        );
        assert_eq!(
            t("[[:lower:]]"),
            hir_ascii_uclass(&ast::ClassAsciiKind::Lower)
        );
        assert_eq!(
            t("[[:print:]]"),
            hir_ascii_uclass(&ast::ClassAsciiKind::Print)
        );
        assert_eq!(
            t("[[:punct:]]"),
            hir_ascii_uclass(&ast::ClassAsciiKind::Punct)
        );
        assert_eq!(
            t("[[:space:]]"),
            hir_ascii_uclass(&ast::ClassAsciiKind::Space)
        );
        assert_eq!(
            t("[[:upper:]]"),
            hir_ascii_uclass(&ast::ClassAsciiKind::Upper)
        );
        assert_eq!(
            t("[[:word:]]"),
            hir_ascii_uclass(&ast::ClassAsciiKind::Word)
        );
        assert_eq!(
            t("[[:xdigit:]]"),
            hir_ascii_uclass(&ast::ClassAsciiKind::Xdigit)
        );

        assert_eq!(
            t("[[:^lower:]]"),
            hir_negate(hir_ascii_uclass(&ast::ClassAsciiKind::Lower))
        );
        #[cfg(feature = "unicode-case")]
        assert_eq!(
            t("(?i)[[:lower:]]"),
            hir_uclass(&[
                ('A', 'Z'),
                ('a', 'z'),
                ('\u{17F}', '\u{17F}'),
                ('\u{212A}', '\u{212A}'),
            ])
        );

        assert_eq!(
            t("(?-u)[[:lower:]]"),
            hir_ascii_bclass(&ast::ClassAsciiKind::Lower)
        );
        assert_eq!(
            t("(?i-u)[[:lower:]]"),
            hir_case_fold(hir_ascii_bclass(&ast::ClassAsciiKind::Lower))
        );

        assert_eq!(
            t_err("(?-u)[[:^lower:]]"),
            TestError {
                kind: hir::ErrorKind::InvalidUtf8,
                span: Span::new(
                    Position::new(6, 1, 7),
                    Position::new(16, 1, 17)
                ),
            }
        );
        assert_eq!(
            t_err("(?i-u)[[:^lower:]]"),
            TestError {
                kind: hir::ErrorKind::InvalidUtf8,
                span: Span::new(
                    Position::new(7, 1, 8),
                    Position::new(17, 1, 18)
                ),
            }
        );
    }

    #[test]
    fn class_ascii_multiple() {
        // See: https://github.com/rust-lang/regex/issues/680
        assert_eq!(
            t("[[:alnum:][:^ascii:]]"),
            hir_union(
                hir_ascii_uclass(&ast::ClassAsciiKind::Alnum),
                hir_uclass(&[('\u{80}', '\u{10FFFF}')]),
            ),
        );
        assert_eq!(
            t_bytes("(?-u)[[:alnum:][:^ascii:]]"),
            hir_union(
                hir_ascii_bclass(&ast::ClassAsciiKind::Alnum),
                hir_bclass(&[(0x80, 0xFF)]),
            ),
        );
    }

    #[test]
    #[cfg(feature = "unicode-perl")]
    fn class_perl_unicode() {
        // Unicode
        assert_eq!(t(r"\d"), hir_uclass_query(ClassQuery::Binary("digit")));
        assert_eq!(t(r"\s"), hir_uclass_query(ClassQuery::Binary("space")));
        assert_eq!(t(r"\w"), hir_uclass_perl_word());
        #[cfg(feature = "unicode-case")]
        assert_eq!(
            t(r"(?i)\d"),
            hir_uclass_query(ClassQuery::Binary("digit"))
        );
        #[cfg(feature = "unicode-case")]
        assert_eq!(
            t(r"(?i)\s"),
            hir_uclass_query(ClassQuery::Binary("space"))
        );
        #[cfg(feature = "unicode-case")]
        assert_eq!(t(r"(?i)\w"), hir_uclass_perl_word());

        // Unicode, negated
        assert_eq!(
            t(r"\D"),
            hir_negate(hir_uclass_query(ClassQuery::Binary("digit")))
        );
        assert_eq!(
            t(r"\S"),
            hir_negate(hir_uclass_query(ClassQuery::Binary("space")))
        );
        assert_eq!(t(r"\W"), hir_negate(hir_uclass_perl_word()));
        #[cfg(feature = "unicode-case")]
        assert_eq!(
            t(r"(?i)\D"),
            hir_negate(hir_uclass_query(ClassQuery::Binary("digit")))
        );
        #[cfg(feature = "unicode-case")]
        assert_eq!(
            t(r"(?i)\S"),
            hir_negate(hir_uclass_query(ClassQuery::Binary("space")))
        );
        #[cfg(feature = "unicode-case")]
        assert_eq!(t(r"(?i)\W"), hir_negate(hir_uclass_perl_word()));
    }

    #[test]
    fn class_perl_ascii() {
        // ASCII only
        assert_eq!(
            t(r"(?-u)\d"),
            hir_ascii_bclass(&ast::ClassAsciiKind::Digit)
        );
        assert_eq!(
            t(r"(?-u)\s"),
            hir_ascii_bclass(&ast::ClassAsciiKind::Space)
        );
        assert_eq!(
            t(r"(?-u)\w"),
            hir_ascii_bclass(&ast::ClassAsciiKind::Word)
        );
        assert_eq!(
            t(r"(?i-u)\d"),
            hir_ascii_bclass(&ast::ClassAsciiKind::Digit)
        );
        assert_eq!(
            t(r"(?i-u)\s"),
            hir_ascii_bclass(&ast::ClassAsciiKind::Space)
        );
        assert_eq!(
            t(r"(?i-u)\w"),
            hir_ascii_bclass(&ast::ClassAsciiKind::Word)
        );

        // ASCII only, negated
        assert_eq!(
            t_bytes(r"(?-u)\D"),
            hir_negate(hir_ascii_bclass(&ast::ClassAsciiKind::Digit))
        );
        assert_eq!(
            t_bytes(r"(?-u)\S"),
            hir_negate(hir_ascii_bclass(&ast::ClassAsciiKind::Space))
        );
        assert_eq!(
            t_bytes(r"(?-u)\W"),
            hir_negate(hir_ascii_bclass(&ast::ClassAsciiKind::Word))
        );
        assert_eq!(
            t_bytes(r"(?i-u)\D"),
            hir_negate(hir_ascii_bclass(&ast::ClassAsciiKind::Digit))
        );
        assert_eq!(
            t_bytes(r"(?i-u)\S"),
            hir_negate(hir_ascii_bclass(&ast::ClassAsciiKind::Space))
        );
        assert_eq!(
            t_bytes(r"(?i-u)\W"),
            hir_negate(hir_ascii_bclass(&ast::ClassAsciiKind::Word))
        );

        // ASCII only, negated, with UTF-8 mode enabled.
        // In this case, negating any Perl class results in an error because
        // all such classes can match invalid UTF-8.
        assert_eq!(
            t_err(r"(?-u)\D"),
            TestError {
                kind: hir::ErrorKind::InvalidUtf8,
                span: Span::new(
                    Position::new(5, 1, 6),
                    Position::new(7, 1, 8),
                ),
            },
        );
        assert_eq!(
            t_err(r"(?-u)\S"),
            TestError {
                kind: hir::ErrorKind::InvalidUtf8,
                span: Span::new(
                    Position::new(5, 1, 6),
                    Position::new(7, 1, 8),
                ),
            },
        );
        assert_eq!(
            t_err(r"(?-u)\W"),
            TestError {
                kind: hir::ErrorKind::InvalidUtf8,
                span: Span::new(
                    Position::new(5, 1, 6),
                    Position::new(7, 1, 8),
                ),
            },
        );
        assert_eq!(
            t_err(r"(?i-u)\D"),
            TestError {
                kind: hir::ErrorKind::InvalidUtf8,
                span: Span::new(
                    Position::new(6, 1, 7),
                    Position::new(8, 1, 9),
                ),
            },
        );
        assert_eq!(
            t_err(r"(?i-u)\S"),
            TestError {
                kind: hir::ErrorKind::InvalidUtf8,
                span: Span::new(
                    Position::new(6, 1, 7),
                    Position::new(8, 1, 9),
                ),
            },
        );
        assert_eq!(
            t_err(r"(?i-u)\W"),
            TestError {
                kind: hir::ErrorKind::InvalidUtf8,
                span: Span::new(
                    Position::new(6, 1, 7),
                    Position::new(8, 1, 9),
                ),
            },
        );
    }

    #[test]
    #[cfg(not(feature = "unicode-perl"))]
    fn class_perl_word_disabled() {
        assert_eq!(
            t_err(r"\w"),
            TestError {
                kind: hir::ErrorKind::UnicodePerlClassNotFound,
                span: Span::new(
                    Position::new(0, 1, 1),
                    Position::new(2, 1, 3)
                ),
            }
        );
    }

    #[test]
    #[cfg(all(not(feature = "unicode-perl"), not(feature = "unicode-bool")))]
    fn class_perl_space_disabled() {
        assert_eq!(
            t_err(r"\s"),
            TestError {
                kind: hir::ErrorKind::UnicodePerlClassNotFound,
                span: Span::new(
                    Position::new(0, 1, 1),
                    Position::new(2, 1, 3)
                ),
            }
        );
    }

    #[test]
    #[cfg(all(
        not(feature = "unicode-perl"),
        not(feature = "unicode-gencat")
    ))]
    fn class_perl_digit_disabled() {
        assert_eq!(
            t_err(r"\d"),
            TestError {
                kind: hir::ErrorKind::UnicodePerlClassNotFound,
                span: Span::new(
                    Position::new(0, 1, 1),
                    Position::new(2, 1, 3)
                ),
            }
        );
    }

    #[test]
    #[cfg(feature = "unicode-gencat")]
    fn class_unicode_gencat() {
        assert_eq!(t(r"\pZ"), hir_uclass_query(ClassQuery::Binary("Z")));
        assert_eq!(t(r"\pz"), hir_uclass_query(ClassQuery::Binary("Z")));
        assert_eq!(
            t(r"\p{Separator}"),
            hir_uclass_query(ClassQuery::Binary("Z"))
        );
        assert_eq!(
            t(r"\p{se      PaRa ToR}"),
            hir_uclass_query(ClassQuery::Binary("Z"))
        );
        assert_eq!(
            t(r"\p{gc:Separator}"),
            hir_uclass_query(ClassQuery::Binary("Z"))
        );
        assert_eq!(
            t(r"\p{gc=Separator}"),
            hir_uclass_query(ClassQuery::Binary("Z"))
        );
        assert_eq!(
            t(r"\p{Other}"),
            hir_uclass_query(ClassQuery::Binary("Other"))
        );
        assert_eq!(t(r"\pC"), hir_uclass_query(ClassQuery::Binary("Other")));

        assert_eq!(
            t(r"\PZ"),
            hir_negate(hir_uclass_query(ClassQuery::Binary("Z")))
        );
        assert_eq!(
            t(r"\P{separator}"),
            hir_negate(hir_uclass_query(ClassQuery::Binary("Z")))
        );
        assert_eq!(
            t(r"\P{gc!=separator}"),
            hir_negate(hir_uclass_query(ClassQuery::Binary("Z")))
        );

        assert_eq!(t(r"\p{any}"), hir_uclass_query(ClassQuery::Binary("Any")));
        assert_eq!(
            t(r"\p{assigned}"),
            hir_uclass_query(ClassQuery::Binary("Assigned"))
        );
        assert_eq!(
            t(r"\p{ascii}"),
            hir_uclass_query(ClassQuery::Binary("ASCII"))
        );
        assert_eq!(
            t(r"\p{gc:any}"),
            hir_uclass_query(ClassQuery::Binary("Any"))
        );
        assert_eq!(
            t(r"\p{gc:assigned}"),
            hir_uclass_query(ClassQuery::Binary("Assigned"))
        );
        assert_eq!(
            t(r"\p{gc:ascii}"),
            hir_uclass_query(ClassQuery::Binary("ASCII"))
        );

        assert_eq!(
            t_err(r"(?-u)\pZ"),
            TestError {
                kind: hir::ErrorKind::UnicodeNotAllowed,
                span: Span::new(
                    Position::new(5, 1, 6),
                    Position::new(8, 1, 9)
                ),
            }
        );
        assert_eq!(
            t_err(r"(?-u)\p{Separator}"),
            TestError {
                kind: hir::ErrorKind::UnicodeNotAllowed,
                span: Span::new(
                    Position::new(5, 1, 6),
                    Position::new(18, 1, 19)
                ),
            }
        );
        assert_eq!(
            t_err(r"\pE"),
            TestError {
                kind: hir::ErrorKind::UnicodePropertyNotFound,
                span: Span::new(
                    Position::new(0, 1, 1),
                    Position::new(3, 1, 4)
                ),
            }
        );
        assert_eq!(
            t_err(r"\p{Foo}"),
            TestError {
                kind: hir::ErrorKind::UnicodePropertyNotFound,
                span: Span::new(
                    Position::new(0, 1, 1),
                    Position::new(7, 1, 8)
                ),
            }
        );
        assert_eq!(
            t_err(r"\p{gc:Foo}"),
            TestError {
                kind: hir::ErrorKind::UnicodePropertyValueNotFound,
                span: Span::new(
                    Position::new(0, 1, 1),
                    Position::new(10, 1, 11)
                ),
            }
        );
    }

    #[test]
    #[cfg(not(feature = "unicode-gencat"))]
    fn class_unicode_gencat_disabled() {
        assert_eq!(
            t_err(r"\p{Separator}"),
            TestError {
                kind: hir::ErrorKind::UnicodePropertyNotFound,
                span: Span::new(
                    Position::new(0, 1, 1),
                    Position::new(13, 1, 14)
                ),
            }
        );

        assert_eq!(
            t_err(r"\p{Any}"),
            TestError {
                kind: hir::ErrorKind::UnicodePropertyNotFound,
                span: Span::new(
                    Position::new(0, 1, 1),
                    Position::new(7, 1, 8)
                ),
            }
        );
    }

    #[test]
    #[cfg(feature = "unicode-script")]
    fn class_unicode_script() {
        assert_eq!(
            t(r"\p{Greek}"),
            hir_uclass_query(ClassQuery::Binary("Greek"))
        );
        #[cfg(feature = "unicode-case")]
        assert_eq!(
            t(r"(?i)\p{Greek}"),
            hir_case_fold(hir_uclass_query(ClassQuery::Binary("Greek")))
        );
        #[cfg(feature = "unicode-case")]
        assert_eq!(
            t(r"(?i)\P{Greek}"),
            hir_negate(hir_case_fold(hir_uclass_query(ClassQuery::Binary(
                "Greek"
            ))))
        );

        assert_eq!(
            t_err(r"\p{sc:Foo}"),
            TestError {
                kind: hir::ErrorKind::UnicodePropertyValueNotFound,
                span: Span::new(
                    Position::new(0, 1, 1),
                    Position::new(10, 1, 11)
                ),
            }
        );
        assert_eq!(
            t_err(r"\p{scx:Foo}"),
            TestError {
                kind: hir::ErrorKind::UnicodePropertyValueNotFound,
                span: Span::new(
                    Position::new(0, 1, 1),
                    Position::new(11, 1, 12)
                ),
            }
        );
    }

    #[test]
    #[cfg(not(feature = "unicode-script"))]
    fn class_unicode_script_disabled() {
        assert_eq!(
            t_err(r"\p{Greek}"),
            TestError {
                kind: hir::ErrorKind::UnicodePropertyNotFound,
                span: Span::new(
                    Position::new(0, 1, 1),
                    Position::new(9, 1, 10)
                ),
            }
        );

        assert_eq!(
            t_err(r"\p{scx:Greek}"),
            TestError {
                kind: hir::ErrorKind::UnicodePropertyNotFound,
                span: Span::new(
                    Position::new(0, 1, 1),
                    Position::new(13, 1, 14)
                ),
            }
        );
    }

    #[test]
    #[cfg(feature = "unicode-age")]
    fn class_unicode_age() {
        assert_eq!(
            t_err(r"\p{age:Foo}"),
            TestError {
                kind: hir::ErrorKind::UnicodePropertyValueNotFound,
                span: Span::new(
                    Position::new(0, 1, 1),
                    Position::new(11, 1, 12)
                ),
            }
        );
    }

    #[test]
    #[cfg(feature = "unicode-gencat")]
    fn class_unicode_any_empty() {
        assert_eq!(t(r"\P{any}"), hir_uclass(&[]),);
    }

    #[test]
    #[cfg(not(feature = "unicode-age"))]
    fn class_unicode_age_disabled() {
        assert_eq!(
            t_err(r"\p{age:3.0}"),
            TestError {
                kind: hir::ErrorKind::UnicodePropertyNotFound,
                span: Span::new(
                    Position::new(0, 1, 1),
                    Position::new(11, 1, 12)
                ),
            }
        );
    }

    #[test]
    fn class_bracketed() {
        assert_eq!(t("[a]"), hir_lit("a"));
        assert_eq!(t("[ab]"), hir_uclass(&[('a', 'b')]));
        assert_eq!(t("[^[a]]"), class_negate(uclass(&[('a', 'a')])));
        assert_eq!(t("[a-z]"), hir_uclass(&[('a', 'z')]));
        assert_eq!(t("[a-fd-h]"), hir_uclass(&[('a', 'h')]));
        assert_eq!(t("[a-fg-m]"), hir_uclass(&[('a', 'm')]));
        assert_eq!(t(r"[\x00]"), hir_uclass(&[('\0', '\0')]));
        assert_eq!(t(r"[\n]"), hir_uclass(&[('\n', '\n')]));
        assert_eq!(t("[\n]"), hir_uclass(&[('\n', '\n')]));
        #[cfg(any(feature = "unicode-perl", feature = "unicode-gencat"))]
        assert_eq!(t(r"[\d]"), hir_uclass_query(ClassQuery::Binary("digit")));
        #[cfg(feature = "unicode-gencat")]
        assert_eq!(
            t(r"[\pZ]"),
            hir_uclass_query(ClassQuery::Binary("separator"))
        );
        #[cfg(feature = "unicode-gencat")]
        assert_eq!(
            t(r"[\p{separator}]"),
            hir_uclass_query(ClassQuery::Binary("separator"))
        );
        #[cfg(any(feature = "unicode-perl", feature = "unicode-gencat"))]
        assert_eq!(t(r"[^\D]"), hir_uclass_query(ClassQuery::Binary("digit")));
        #[cfg(feature = "unicode-gencat")]
        assert_eq!(
            t(r"[^\PZ]"),
            hir_uclass_query(ClassQuery::Binary("separator"))
        );
        #[cfg(feature = "unicode-gencat")]
        assert_eq!(
            t(r"[^\P{separator}]"),
            hir_uclass_query(ClassQuery::Binary("separator"))
        );
        #[cfg(all(
            feature = "unicode-case",
            any(feature = "unicode-perl", feature = "unicode-gencat")
        ))]
        assert_eq!(
            t(r"(?i)[^\D]"),
            hir_uclass_query(ClassQuery::Binary("digit"))
        );
        #[cfg(all(feature = "unicode-case", feature = "unicode-script"))]
        assert_eq!(
            t(r"(?i)[^\P{greek}]"),
            hir_case_fold(hir_uclass_query(ClassQuery::Binary("greek")))
        );

        assert_eq!(t("(?-u)[a]"), hir_bclass(&[(b'a', b'a')]));
        assert_eq!(t(r"(?-u)[\x00]"), hir_bclass(&[(b'\0', b'\0')]));
        assert_eq!(t_bytes(r"(?-u)[\xFF]"), hir_bclass(&[(b'\xFF', b'\xFF')]));

        #[cfg(feature = "unicode-case")]
        assert_eq!(t("(?i)[a]"), hir_uclass(&[('A', 'A'), ('a', 'a')]));
        #[cfg(feature = "unicode-case")]
        assert_eq!(
            t("(?i)[k]"),
            hir_uclass(&[('K', 'K'), ('k', 'k'), ('\u{212A}', '\u{212A}'),])
        );
        #[cfg(feature = "unicode-case")]
        assert_eq!(
            t("(?i)[β]"),
            hir_uclass(&[('Β', 'Β'), ('β', 'β'), ('ϐ', 'ϐ'),])
        );
        assert_eq!(t("(?i-u)[k]"), hir_bclass(&[(b'K', b'K'), (b'k', b'k'),]));

        assert_eq!(t("[^a]"), class_negate(uclass(&[('a', 'a')])));
        assert_eq!(t(r"[^\x00]"), class_negate(uclass(&[('\0', '\0')])));
        assert_eq!(
            t_bytes("(?-u)[^a]"),
            class_negate(bclass(&[(b'a', b'a')]))
        );
        #[cfg(any(feature = "unicode-perl", feature = "unicode-gencat"))]
        assert_eq!(
            t(r"[^\d]"),
            hir_negate(hir_uclass_query(ClassQuery::Binary("digit")))
        );
        #[cfg(feature = "unicode-gencat")]
        assert_eq!(
            t(r"[^\pZ]"),
            hir_negate(hir_uclass_query(ClassQuery::Binary("separator")))
        );
        #[cfg(feature = "unicode-gencat")]
        assert_eq!(
            t(r"[^\p{separator}]"),
            hir_negate(hir_uclass_query(ClassQuery::Binary("separator")))
        );
        #[cfg(all(feature = "unicode-case", feature = "unicode-script"))]
        assert_eq!(
            t(r"(?i)[^\p{greek}]"),
            hir_negate(hir_case_fold(hir_uclass_query(ClassQuery::Binary(
                "greek"
            ))))
        );
        #[cfg(all(feature = "unicode-case", feature = "unicode-script"))]
        assert_eq!(
            t(r"(?i)[\P{greek}]"),
            hir_negate(hir_case_fold(hir_uclass_query(ClassQuery::Binary(
                "greek"
            ))))
        );

        // Test some weird cases.
        assert_eq!(t(r"[\[]"), hir_uclass(&[('[', '[')]));

        assert_eq!(t(r"[&]"), hir_uclass(&[('&', '&')]));
        assert_eq!(t(r"[\&]"), hir_uclass(&[('&', '&')]));
        assert_eq!(t(r"[\&\&]"), hir_uclass(&[('&', '&')]));
        assert_eq!(t(r"[\x00-&]"), hir_uclass(&[('\0', '&')]));
        assert_eq!(t(r"[&-\xFF]"), hir_uclass(&[('&', '\u{FF}')]));

        assert_eq!(t(r"[~]"), hir_uclass(&[('~', '~')]));
        assert_eq!(t(r"[\~]"), hir_uclass(&[('~', '~')]));
        assert_eq!(t(r"[\~\~]"), hir_uclass(&[('~', '~')]));
        assert_eq!(t(r"[\x00-~]"), hir_uclass(&[('\0', '~')]));
        assert_eq!(t(r"[~-\xFF]"), hir_uclass(&[('~', '\u{FF}')]));

        assert_eq!(t(r"[-]"), hir_uclass(&[('-', '-')]));
        assert_eq!(t(r"[\-]"), hir_uclass(&[('-', '-')]));
        assert_eq!(t(r"[\-\-]"), hir_uclass(&[('-', '-')]));
        assert_eq!(t(r"[\x00-\-]"), hir_uclass(&[('\0', '-')]));
        assert_eq!(t(r"[\--\xFF]"), hir_uclass(&[('-', '\u{FF}')]));

        assert_eq!(
            t_err("(?-u)[^a]"),
            TestError {
                kind: hir::ErrorKind::InvalidUtf8,
                span: Span::new(
                    Position::new(5, 1, 6),
                    Position::new(9, 1, 10)
                ),
            }
        );
        #[cfg(any(feature = "unicode-perl", feature = "unicode-bool"))]
        assert_eq!(t(r"[^\s\S]"), hir_uclass(&[]),);
        #[cfg(any(feature = "unicode-perl", feature = "unicode-bool"))]
        assert_eq!(t_bytes(r"(?-u)[^\s\S]"), hir_bclass(&[]),);
    }

    #[test]
    fn class_bracketed_union() {
        assert_eq!(t("[a-zA-Z]"), hir_uclass(&[('A', 'Z'), ('a', 'z')]));
        #[cfg(feature = "unicode-gencat")]
        assert_eq!(
            t(r"[a\pZb]"),
            hir_union(
                hir_uclass(&[('a', 'b')]),
                hir_uclass_query(ClassQuery::Binary("separator"))
            )
        );
        #[cfg(all(feature = "unicode-gencat", feature = "unicode-script"))]
        assert_eq!(
            t(r"[\pZ\p{Greek}]"),
            hir_union(
                hir_uclass_query(ClassQuery::Binary("greek")),
                hir_uclass_query(ClassQuery::Binary("separator"))
            )
        );
        #[cfg(all(
            feature = "unicode-age",
            feature = "unicode-gencat",
            feature = "unicode-script"
        ))]
        assert_eq!(
            t(r"[\p{age:3.0}\pZ\p{Greek}]"),
            hir_union(
                hir_uclass_query(ClassQuery::ByValue {
                    property_name: "age",
                    property_value: "3.0",
                }),
                hir_union(
                    hir_uclass_query(ClassQuery::Binary("greek")),
                    hir_uclass_query(ClassQuery::Binary("separator"))
                )
            )
        );
        #[cfg(all(
            feature = "unicode-age",
            feature = "unicode-gencat",
            feature = "unicode-script"
        ))]
        assert_eq!(
            t(r"[[[\p{age:3.0}\pZ]\p{Greek}][\p{Cyrillic}]]"),
            hir_union(
                hir_uclass_query(ClassQuery::ByValue {
                    property_name: "age",
                    property_value: "3.0",
                }),
                hir_union(
                    hir_uclass_query(ClassQuery::Binary("cyrillic")),
                    hir_union(
                        hir_uclass_query(ClassQuery::Binary("greek")),
                        hir_uclass_query(ClassQuery::Binary("separator"))
                    )
                )
            )
        );

        #[cfg(all(
            feature = "unicode-age",
            feature = "unicode-case",
            feature = "unicode-gencat",
            feature = "unicode-script"
        ))]
        assert_eq!(
            t(r"(?i)[\p{age:3.0}\pZ\p{Greek}]"),
            hir_case_fold(hir_union(
                hir_uclass_query(ClassQuery::ByValue {
                    property_name: "age",
                    property_value: "3.0",
                }),
                hir_union(
                    hir_uclass_query(ClassQuery::Binary("greek")),
                    hir_uclass_query(ClassQuery::Binary("separator"))
                )
            ))
        );
        #[cfg(all(
            feature = "unicode-age",
            feature = "unicode-gencat",
            feature = "unicode-script"
        ))]
        assert_eq!(
            t(r"[^\p{age:3.0}\pZ\p{Greek}]"),
            hir_negate(hir_union(
                hir_uclass_query(ClassQuery::ByValue {
                    property_name: "age",
                    property_value: "3.0",
                }),
                hir_union(
                    hir_uclass_query(ClassQuery::Binary("greek")),
                    hir_uclass_query(ClassQuery::Binary("separator"))
                )
            ))
        );
        #[cfg(all(
            feature = "unicode-age",
            feature = "unicode-case",
            feature = "unicode-gencat",
            feature = "unicode-script"
        ))]
        assert_eq!(
            t(r"(?i)[^\p{age:3.0}\pZ\p{Greek}]"),
            hir_negate(hir_case_fold(hir_union(
                hir_uclass_query(ClassQuery::ByValue {
                    property_name: "age",
                    property_value: "3.0",
                }),
                hir_union(
                    hir_uclass_query(ClassQuery::Binary("greek")),
                    hir_uclass_query(ClassQuery::Binary("separator"))
                )
            )))
        );
    }

    #[test]
    fn class_bracketed_nested() {
        assert_eq!(t(r"[a[^c]]"), class_negate(uclass(&[('c', 'c')])));
        assert_eq!(t(r"[a-b[^c]]"), class_negate(uclass(&[('c', 'c')])));
        assert_eq!(t(r"[a-c[^c]]"), class_negate(uclass(&[])));

        assert_eq!(t(r"[^a[^c]]"), hir_uclass(&[('c', 'c')]));
        assert_eq!(t(r"[^a-b[^c]]"), hir_uclass(&[('c', 'c')]));

        #[cfg(feature = "unicode-case")]
        assert_eq!(
            t(r"(?i)[a[^c]]"),
            hir_negate(class_case_fold(uclass(&[('c', 'c')])))
        );
        #[cfg(feature = "unicode-case")]
        assert_eq!(
            t(r"(?i)[a-b[^c]]"),
            hir_negate(class_case_fold(uclass(&[('c', 'c')])))
        );

        #[cfg(feature = "unicode-case")]
        assert_eq!(t(r"(?i)[^a[^c]]"), hir_uclass(&[('C', 'C'), ('c', 'c')]));
        #[cfg(feature = "unicode-case")]
        assert_eq!(
            t(r"(?i)[^a-b[^c]]"),
            hir_uclass(&[('C', 'C'), ('c', 'c')])
        );

        assert_eq!(t(r"[^a-c[^c]]"), hir_uclass(&[]),);
        #[cfg(feature = "unicode-case")]
        assert_eq!(t(r"(?i)[^a-c[^c]]"), hir_uclass(&[]),);
    }

    #[test]
    fn class_bracketed_intersect() {
        assert_eq!(t("[abc&&b-c]"), hir_uclass(&[('b', 'c')]));
        assert_eq!(t("[abc&&[b-c]]"), hir_uclass(&[('b', 'c')]));
        assert_eq!(t("[[abc]&&[b-c]]"), hir_uclass(&[('b', 'c')]));
        assert_eq!(t("[a-z&&b-y&&c-x]"), hir_uclass(&[('c', 'x')]));
        assert_eq!(t("[c-da-b&&a-d]"), hir_uclass(&[('a', 'd')]));
        assert_eq!(t("[a-d&&c-da-b]"), hir_uclass(&[('a', 'd')]));
        assert_eq!(t(r"[a-z&&a-c]"), hir_uclass(&[('a', 'c')]));
        assert_eq!(t(r"[[a-z&&a-c]]"), hir_uclass(&[('a', 'c')]));
        assert_eq!(t(r"[^[a-z&&a-c]]"), hir_negate(hir_uclass(&[('a', 'c')])));

        assert_eq!(t("(?-u)[abc&&b-c]"), hir_bclass(&[(b'b', b'c')]));
        assert_eq!(t("(?-u)[abc&&[b-c]]"), hir_bclass(&[(b'b', b'c')]));
        assert_eq!(t("(?-u)[[abc]&&[b-c]]"), hir_bclass(&[(b'b', b'c')]));
        assert_eq!(t("(?-u)[a-z&&b-y&&c-x]"), hir_bclass(&[(b'c', b'x')]));
        assert_eq!(t("(?-u)[c-da-b&&a-d]"), hir_bclass(&[(b'a', b'd')]));
        assert_eq!(t("(?-u)[a-d&&c-da-b]"), hir_bclass(&[(b'a', b'd')]));

        #[cfg(feature = "unicode-case")]
        assert_eq!(
            t("(?i)[abc&&b-c]"),
            hir_case_fold(hir_uclass(&[('b', 'c')]))
        );
        #[cfg(feature = "unicode-case")]
        assert_eq!(
            t("(?i)[abc&&[b-c]]"),
            hir_case_fold(hir_uclass(&[('b', 'c')]))
        );
        #[cfg(feature = "unicode-case")]
        assert_eq!(
            t("(?i)[[abc]&&[b-c]]"),
            hir_case_fold(hir_uclass(&[('b', 'c')]))
        );
        #[cfg(feature = "unicode-case")]
        assert_eq!(
            t("(?i)[a-z&&b-y&&c-x]"),
            hir_case_fold(hir_uclass(&[('c', 'x')]))
        );
        #[cfg(feature = "unicode-case")]
        assert_eq!(
            t("(?i)[c-da-b&&a-d]"),
            hir_case_fold(hir_uclass(&[('a', 'd')]))
        );
        #[cfg(feature = "unicode-case")]
        assert_eq!(
            t("(?i)[a-d&&c-da-b]"),
            hir_case_fold(hir_uclass(&[('a', 'd')]))
        );

        assert_eq!(
            t("(?i-u)[abc&&b-c]"),
            hir_case_fold(hir_bclass(&[(b'b', b'c')]))
        );
        assert_eq!(
            t("(?i-u)[abc&&[b-c]]"),
            hir_case_fold(hir_bclass(&[(b'b', b'c')]))
        );
        assert_eq!(
            t("(?i-u)[[abc]&&[b-c]]"),
            hir_case_fold(hir_bclass(&[(b'b', b'c')]))
        );
        assert_eq!(
            t("(?i-u)[a-z&&b-y&&c-x]"),
            hir_case_fold(hir_bclass(&[(b'c', b'x')]))
        );
        assert_eq!(
            t("(?i-u)[c-da-b&&a-d]"),
            hir_case_fold(hir_bclass(&[(b'a', b'd')]))
        );
        assert_eq!(
            t("(?i-u)[a-d&&c-da-b]"),
            hir_case_fold(hir_bclass(&[(b'a', b'd')]))
        );

        // In `[a^]`, `^` does not need to be escaped, so it makes sense that
        // `^` is also allowed to be unescaped after `&&`.
        assert_eq!(t(r"[\^&&^]"), hir_uclass(&[('^', '^')]));
        // `]` needs to be escaped after `&&` since it's not at start of class.
        assert_eq!(t(r"[]&&\]]"), hir_uclass(&[(']', ']')]));
        assert_eq!(t(r"[-&&-]"), hir_uclass(&[('-', '-')]));
        assert_eq!(t(r"[\&&&&]"), hir_uclass(&[('&', '&')]));
        assert_eq!(t(r"[\&&&\&]"), hir_uclass(&[('&', '&')]));
        // Test precedence.
        assert_eq!(
            t(r"[a-w&&[^c-g]z]"),
            hir_uclass(&[('a', 'b'), ('h', 'w')])
        );
    }

    #[test]
    fn class_bracketed_intersect_negate() {
        #[cfg(feature = "unicode-perl")]
        assert_eq!(
            t(r"[^\w&&\d]"),
            hir_negate(hir_uclass_query(ClassQuery::Binary("digit")))
        );
        assert_eq!(t(r"[^[a-z&&a-c]]"), hir_negate(hir_uclass(&[('a', 'c')])));
        #[cfg(feature = "unicode-perl")]
        assert_eq!(
            t(r"[^[\w&&\d]]"),
            hir_negate(hir_uclass_query(ClassQuery::Binary("digit")))
        );
        #[cfg(feature = "unicode-perl")]
        assert_eq!(
            t(r"[^[^\w&&\d]]"),
            hir_uclass_query(ClassQuery::Binary("digit"))
        );
        #[cfg(feature = "unicode-perl")]
        assert_eq!(t(r"[[[^\w]&&[^\d]]]"), hir_negate(hir_uclass_perl_word()));

        #[cfg(feature = "unicode-perl")]
        assert_eq!(
            t_bytes(r"(?-u)[^\w&&\d]"),
            hir_negate(hir_ascii_bclass(&ast::ClassAsciiKind::Digit))
        );
        assert_eq!(
            t_bytes(r"(?-u)[^[a-z&&a-c]]"),
            hir_negate(hir_bclass(&[(b'a', b'c')]))
        );
        assert_eq!(
            t_bytes(r"(?-u)[^[\w&&\d]]"),
            hir_negate(hir_ascii_bclass(&ast::ClassAsciiKind::Digit))
        );
        assert_eq!(
            t_bytes(r"(?-u)[^[^\w&&\d]]"),
            hir_ascii_bclass(&ast::ClassAsciiKind::Digit)
        );
        assert_eq!(
            t_bytes(r"(?-u)[[[^\w]&&[^\d]]]"),
            hir_negate(hir_ascii_bclass(&ast::ClassAsciiKind::Word))
        );
    }

    #[test]
    fn class_bracketed_difference() {
        #[cfg(feature = "unicode-gencat")]
        assert_eq!(
            t(r"[\pL--[:ascii:]]"),
            hir_difference(
                hir_uclass_query(ClassQuery::Binary("letter")),
                hir_uclass(&[('\0', '\x7F')])
            )
        );

        assert_eq!(
            t(r"(?-u)[[:alpha:]--[:lower:]]"),
            hir_bclass(&[(b'A', b'Z')])
        );
    }

    #[test]
    fn class_bracketed_symmetric_difference() {
        #[cfg(feature = "unicode-script")]
        assert_eq!(
            t(r"[\p{sc:Greek}~~\p{scx:Greek}]"),
            // Class({
            //     '·'..='·',
            //     '\u{300}'..='\u{301}',
            //     '\u{304}'..='\u{304}',
            //     '\u{306}'..='\u{306}',
            //     '\u{308}'..='\u{308}',
            //     '\u{313}'..='\u{313}',
            //     '\u{342}'..='\u{342}',
            //     '\u{345}'..='\u{345}',
            //     'ʹ'..='ʹ',
            //     '\u{1dc0}'..='\u{1dc1}',
            //     '⁝'..='⁝',
            // })
            hir_uclass(&[
                ('·', '·'),
                ('\u{0300}', '\u{0301}'),
                ('\u{0304}', '\u{0304}'),
                ('\u{0306}', '\u{0306}'),
                ('\u{0308}', '\u{0308}'),
                ('\u{0313}', '\u{0313}'),
                ('\u{0342}', '\u{0342}'),
                ('\u{0345}', '\u{0345}'),
                ('ʹ', 'ʹ'),
                ('\u{1DC0}', '\u{1DC1}'),
                ('⁝', '⁝'),
            ])
        );
        assert_eq!(t(r"[a-g~~c-j]"), hir_uclass(&[('a', 'b'), ('h', 'j')]));

        assert_eq!(
            t(r"(?-u)[a-g~~c-j]"),
            hir_bclass(&[(b'a', b'b'), (b'h', b'j')])
        );
    }

    #[test]
    fn ignore_whitespace() {
        assert_eq!(t(r"(?x)\12 3"), hir_lit("\n3"));
        assert_eq!(t(r"(?x)\x { 53 }"), hir_lit("S"));
        assert_eq!(
            t(r"(?x)\x # comment
{ # comment
    53 # comment
} #comment"),
            hir_lit("S")
        );

        assert_eq!(t(r"(?x)\x 53"), hir_lit("S"));
        assert_eq!(
            t(r"(?x)\x # comment
        53 # comment"),
            hir_lit("S")
        );
        assert_eq!(t(r"(?x)\x5 3"), hir_lit("S"));

        #[cfg(feature = "unicode-gencat")]
        assert_eq!(
            t(r"(?x)\p # comment
{ # comment
    Separator # comment
} # comment"),
            hir_uclass_query(ClassQuery::Binary("separator"))
        );

        assert_eq!(
            t(r"(?x)a # comment
{ # comment
    5 # comment
    , # comment
    10 # comment
} # comment"),
            hir_range(true, 5, Some(10), hir_lit("a"))
        );

        assert_eq!(t(r"(?x)a\  # hi there"), hir_lit("a "));
    }

    #[test]
    fn analysis_is_utf8() {
        // Positive examples.
        assert!(props_bytes(r"a").is_utf8());
        assert!(props_bytes(r"ab").is_utf8());
        assert!(props_bytes(r"(?-u)a").is_utf8());
        assert!(props_bytes(r"(?-u)ab").is_utf8());
        assert!(props_bytes(r"\xFF").is_utf8());
        assert!(props_bytes(r"\xFF\xFF").is_utf8());
        assert!(props_bytes(r"[^a]").is_utf8());
        assert!(props_bytes(r"[^a][^a]").is_utf8());
        assert!(props_bytes(r"\b").is_utf8());
        assert!(props_bytes(r"\B").is_utf8());
        assert!(props_bytes(r"(?-u)\b").is_utf8());
        assert!(props_bytes(r"(?-u)\B").is_utf8());

        // Negative examples.
        assert!(!props_bytes(r"(?-u)\xFF").is_utf8());
        assert!(!props_bytes(r"(?-u)\xFF\xFF").is_utf8());
        assert!(!props_bytes(r"(?-u)[^a]").is_utf8());
        assert!(!props_bytes(r"(?-u)[^a][^a]").is_utf8());
    }

    #[test]
    fn analysis_captures_len() {
        assert_eq!(0, props(r"a").explicit_captures_len());
        assert_eq!(0, props(r"(?:a)").explicit_captures_len());
        assert_eq!(0, props(r"(?i-u:a)").explicit_captures_len());
        assert_eq!(0, props(r"(?i-u)a").explicit_captures_len());
        assert_eq!(1, props(r"(a)").explicit_captures_len());
        assert_eq!(1, props(r"(?P<foo>a)").explicit_captures_len());
        assert_eq!(1, props(r"()").explicit_captures_len());
        assert_eq!(1, props(r"()a").explicit_captures_len());
        assert_eq!(1, props(r"(a)+").explicit_captures_len());
        assert_eq!(2, props(r"(a)(b)").explicit_captures_len());
        assert_eq!(2, props(r"(a)|(b)").explicit_captures_len());
        assert_eq!(2, props(r"((a))").explicit_captures_len());
        assert_eq!(1, props(r"([a&&b])").explicit_captures_len());
    }

    #[test]
    fn analysis_static_captures_len() {
        let len = |pattern| props(pattern).static_explicit_captures_len();
        assert_eq!(Some(0), len(r""));
        assert_eq!(Some(0), len(r"foo|bar"));
        assert_eq!(None, len(r"(foo)|bar"));
        assert_eq!(None, len(r"foo|(bar)"));
        assert_eq!(Some(1), len(r"(foo|bar)"));
        assert_eq!(Some(1), len(r"(a|b|c|d|e|f)"));
        assert_eq!(Some(1), len(r"(a)|(b)|(c)|(d)|(e)|(f)"));
        assert_eq!(Some(2), len(r"(a)(b)|(c)(d)|(e)(f)"));
        assert_eq!(Some(6), len(r"(a)(b)(c)(d)(e)(f)"));
        assert_eq!(Some(3), len(r"(a)(b)(extra)|(a)(b)()"));
        assert_eq!(Some(3), len(r"(a)(b)((?:extra)?)"));
        assert_eq!(None, len(r"(a)(b)(extra)?"));
        assert_eq!(Some(1), len(r"(foo)|(bar)"));
        assert_eq!(Some(2), len(r"(foo)(bar)"));
        assert_eq!(Some(2), len(r"(foo)+(bar)"));
        assert_eq!(None, len(r"(foo)*(bar)"));
        assert_eq!(Some(0), len(r"(foo)?{0}"));
        assert_eq!(None, len(r"(foo)?{1}"));
        assert_eq!(Some(1), len(r"(foo){1}"));
        assert_eq!(Some(1), len(r"(foo){1,}"));
        assert_eq!(Some(1), len(r"(foo){1,}?"));
        assert_eq!(None, len(r"(foo){1,}??"));
        assert_eq!(None, len(r"(foo){0,}"));
        assert_eq!(Some(1), len(r"(foo)(?:bar)"));
        assert_eq!(Some(2), len(r"(foo(?:bar)+)(?:baz(boo))"));
        assert_eq!(Some(2), len(r"(?P<bar>foo)(?:bar)(bal|loon)"));
        assert_eq!(
            Some(2),
            len(r#"<(a)[^>]+href="([^"]+)"|<(img)[^>]+src="([^"]+)""#)
        );
    }

    #[test]
    fn analysis_is_all_assertions() {
        // Positive examples.
        let p = props(r"\b");
        assert!(!p.look_set().is_empty());
        assert_eq!(p.minimum_len(), Some(0));

        let p = props(r"\B");
        assert!(!p.look_set().is_empty());
        assert_eq!(p.minimum_len(), Some(0));

        let p = props(r"^");
        assert!(!p.look_set().is_empty());
        assert_eq!(p.minimum_len(), Some(0));

        let p = props(r"$");
        assert!(!p.look_set().is_empty());
        assert_eq!(p.minimum_len(), Some(0));

        let p = props(r"\A");
        assert!(!p.look_set().is_empty());
        assert_eq!(p.minimum_len(), Some(0));

        let p = props(r"\z");
        assert!(!p.look_set().is_empty());
        assert_eq!(p.minimum_len(), Some(0));

        let p = props(r"$^\z\A\b\B");
        assert!(!p.look_set().is_empty());
        assert_eq!(p.minimum_len(), Some(0));

        let p = props(r"$|^|\z|\A|\b|\B");
        assert!(!p.look_set().is_empty());
        assert_eq!(p.minimum_len(), Some(0));

        let p = props(r"^$|$^");
        assert!(!p.look_set().is_empty());
        assert_eq!(p.minimum_len(), Some(0));

        let p = props(r"((\b)+())*^");
        assert!(!p.look_set().is_empty());
        assert_eq!(p.minimum_len(), Some(0));

        // Negative examples.
        let p = props(r"^a");
        assert!(!p.look_set().is_empty());
        assert_eq!(p.minimum_len(), Some(1));
    }

    #[test]
    fn analysis_look_set_prefix_any() {
        let p = props(r"(?-u)(?i:(?:\b|_)win(?:32|64|dows)?(?:\b|_))");
        assert!(p.look_set_prefix_any().contains(Look::WordAscii));
    }

    #[test]
    fn analysis_is_anchored() {
        let is_start = |p| props(p).look_set_prefix().contains(Look::Start);
        let is_end = |p| props(p).look_set_suffix().contains(Look::End);

        // Positive examples.
        assert!(is_start(r"^"));
        assert!(is_end(r"$"));

        assert!(is_start(r"^^"));
        assert!(props(r"$$").look_set_suffix().contains(Look::End));

        assert!(is_start(r"^$"));
        assert!(is_end(r"^$"));

        assert!(is_start(r"^foo"));
        assert!(is_end(r"foo$"));

        assert!(is_start(r"^foo|^bar"));
        assert!(is_end(r"foo$|bar$"));

        assert!(is_start(r"^(foo|bar)"));
        assert!(is_end(r"(foo|bar)$"));

        assert!(is_start(r"^+"));
        assert!(is_end(r"$+"));
        assert!(is_start(r"^++"));
        assert!(is_end(r"$++"));
        assert!(is_start(r"(^)+"));
        assert!(is_end(r"($)+"));

        assert!(is_start(r"$^"));
        assert!(is_start(r"$^"));
        assert!(is_start(r"$^|^$"));
        assert!(is_end(r"$^|^$"));

        assert!(is_start(r"\b^"));
        assert!(is_end(r"$\b"));
        assert!(is_start(r"^(?m:^)"));
        assert!(is_end(r"(?m:$)$"));
        assert!(is_start(r"(?m:^)^"));
        assert!(is_end(r"$(?m:$)"));

        // Negative examples.
        assert!(!is_start(r"(?m)^"));
        assert!(!is_end(r"(?m)$"));
        assert!(!is_start(r"(?m:^$)|$^"));
        assert!(!is_end(r"(?m:^$)|$^"));
        assert!(!is_start(r"$^|(?m:^$)"));
        assert!(!is_end(r"$^|(?m:^$)"));

        assert!(!is_start(r"a^"));
        assert!(!is_start(r"$a"));

        assert!(!is_end(r"a^"));
        assert!(!is_end(r"$a"));

        assert!(!is_start(r"^foo|bar"));
        assert!(!is_end(r"foo|bar$"));

        assert!(!is_start(r"^*"));
        assert!(!is_end(r"$*"));
        assert!(!is_start(r"^*+"));
        assert!(!is_end(r"$*+"));
        assert!(!is_start(r"^+*"));
        assert!(!is_end(r"$+*"));
        assert!(!is_start(r"(^)*"));
        assert!(!is_end(r"($)*"));
    }

    #[test]
    fn analysis_is_any_anchored() {
        let is_start = |p| props(p).look_set().contains(Look::Start);
        let is_end = |p| props(p).look_set().contains(Look::End);

        // Positive examples.
        assert!(is_start(r"^"));
        assert!(is_end(r"$"));
        assert!(is_start(r"\A"));
        assert!(is_end(r"\z"));

        // Negative examples.
        assert!(!is_start(r"(?m)^"));
        assert!(!is_end(r"(?m)$"));
        assert!(!is_start(r"$"));
        assert!(!is_end(r"^"));
    }

    #[test]
    fn analysis_can_empty() {
        // Positive examples.
        let assert_empty =
            |p| assert_eq!(Some(0), props_bytes(p).minimum_len());
        assert_empty(r"");
        assert_empty(r"()");
        assert_empty(r"()*");
        assert_empty(r"()+");
        assert_empty(r"()?");
        assert_empty(r"a*");
        assert_empty(r"a?");
        assert_empty(r"a{0}");
        assert_empty(r"a{0,}");
        assert_empty(r"a{0,1}");
        assert_empty(r"a{0,10}");
        #[cfg(feature = "unicode-gencat")]
        assert_empty(r"\pL*");
        assert_empty(r"a*|b");
        assert_empty(r"b|a*");
        assert_empty(r"a|");
        assert_empty(r"|a");
        assert_empty(r"a||b");
        assert_empty(r"a*a?(abcd)*");
        assert_empty(r"^");
        assert_empty(r"$");
        assert_empty(r"(?m)^");
        assert_empty(r"(?m)$");
        assert_empty(r"\A");
        assert_empty(r"\z");
        assert_empty(r"\B");
        assert_empty(r"(?-u)\B");
        assert_empty(r"\b");
        assert_empty(r"(?-u)\b");

        // Negative examples.
        let assert_non_empty =
            |p| assert_ne!(Some(0), props_bytes(p).minimum_len());
        assert_non_empty(r"a+");
        assert_non_empty(r"a{1}");
        assert_non_empty(r"a{1,}");
        assert_non_empty(r"a{1,2}");
        assert_non_empty(r"a{1,10}");
        assert_non_empty(r"b|a");
        assert_non_empty(r"a*a+(abcd)*");
        #[cfg(feature = "unicode-gencat")]
        assert_non_empty(r"\P{any}");
        assert_non_empty(r"[a--a]");
        assert_non_empty(r"[a&&b]");
    }

    #[test]
    fn analysis_is_literal() {
        // Positive examples.
        assert!(props(r"a").is_literal());
        assert!(props(r"ab").is_literal());
        assert!(props(r"abc").is_literal());
        assert!(props(r"(?m)abc").is_literal());
        assert!(props(r"(?:a)").is_literal());
        assert!(props(r"foo(?:a)").is_literal());
        assert!(props(r"(?:a)foo").is_literal());
        assert!(props(r"[a]").is_literal());

        // Negative examples.
        assert!(!props(r"").is_literal());
        assert!(!props(r"^").is_literal());
        assert!(!props(r"a|b").is_literal());
        assert!(!props(r"(a)").is_literal());
        assert!(!props(r"a+").is_literal());
        assert!(!props(r"foo(a)").is_literal());
        assert!(!props(r"(a)foo").is_literal());
        assert!(!props(r"[ab]").is_literal());
    }

    #[test]
    fn analysis_is_alternation_literal() {
        // Positive examples.
        assert!(props(r"a").is_alternation_literal());
        assert!(props(r"ab").is_alternation_literal());
        assert!(props(r"abc").is_alternation_literal());
        assert!(props(r"(?m)abc").is_alternation_literal());
        assert!(props(r"foo|bar").is_alternation_literal());
        assert!(props(r"foo|bar|baz").is_alternation_literal());
        assert!(props(r"[a]").is_alternation_literal());
        assert!(props(r"(?:ab)|cd").is_alternation_literal());
        assert!(props(r"ab|(?:cd)").is_alternation_literal());

        // Negative examples.
        assert!(!props(r"").is_alternation_literal());
        assert!(!props(r"^").is_alternation_literal());
        assert!(!props(r"(a)").is_alternation_literal());
        assert!(!props(r"a+").is_alternation_literal());
        assert!(!props(r"foo(a)").is_alternation_literal());
        assert!(!props(r"(a)foo").is_alternation_literal());
        assert!(!props(r"[ab]").is_alternation_literal());
        assert!(!props(r"[ab]|b").is_alternation_literal());
        assert!(!props(r"a|[ab]").is_alternation_literal());
        assert!(!props(r"(a)|b").is_alternation_literal());
        assert!(!props(r"a|(b)").is_alternation_literal());
        assert!(!props(r"a|b").is_alternation_literal());
        assert!(!props(r"a|b|c").is_alternation_literal());
        assert!(!props(r"[a]|b").is_alternation_literal());
        assert!(!props(r"a|[b]").is_alternation_literal());
        assert!(!props(r"(?:a)|b").is_alternation_literal());
        assert!(!props(r"a|(?:b)").is_alternation_literal());
        assert!(!props(r"(?:z|xx)@|xx").is_alternation_literal());
    }

    // This tests that the smart Hir::repetition constructors does some basic
    // simplifications.
    #[test]
    fn smart_repetition() {
        assert_eq!(t(r"a{0}"), Hir::empty());
        assert_eq!(t(r"a{1}"), hir_lit("a"));
        assert_eq!(t(r"\B{32111}"), hir_look(hir::Look::WordUnicodeNegate));
    }

    // This tests that the smart Hir::concat constructor simplifies the given
    // exprs in a way we expect.
    #[test]
    fn smart_concat() {
        assert_eq!(t(""), Hir::empty());
        assert_eq!(t("(?:)"), Hir::empty());
        assert_eq!(t("abc"), hir_lit("abc"));
        assert_eq!(t("(?:foo)(?:bar)"), hir_lit("foobar"));
        assert_eq!(t("quux(?:foo)(?:bar)baz"), hir_lit("quuxfoobarbaz"));
        assert_eq!(
            t("foo(?:bar^baz)quux"),
            hir_cat(vec![
                hir_lit("foobar"),
                hir_look(hir::Look::Start),
                hir_lit("bazquux"),
            ])
        );
        assert_eq!(
            t("foo(?:ba(?:r^b)az)quux"),
            hir_cat(vec![
                hir_lit("foobar"),
                hir_look(hir::Look::Start),
                hir_lit("bazquux"),
            ])
        );
    }

    // This tests that the smart Hir::alternation constructor simplifies the
    // given exprs in a way we expect.
    #[test]
    fn smart_alternation() {
        assert_eq!(
            t("(?:foo)|(?:bar)"),
            hir_alt(vec![hir_lit("foo"), hir_lit("bar")])
        );
        assert_eq!(
            t("quux|(?:abc|def|xyz)|baz"),
            hir_alt(vec![
                hir_lit("quux"),
                hir_lit("abc"),
                hir_lit("def"),
                hir_lit("xyz"),
                hir_lit("baz"),
            ])
        );
        assert_eq!(
            t("quux|(?:abc|(?:def|mno)|xyz)|baz"),
            hir_alt(vec![
                hir_lit("quux"),
                hir_lit("abc"),
                hir_lit("def"),
                hir_lit("mno"),
                hir_lit("xyz"),
                hir_lit("baz"),
            ])
        );
        assert_eq!(
            t("a|b|c|d|e|f|x|y|z"),
            hir_uclass(&[('a', 'f'), ('x', 'z')]),
        );
        // Tests that we lift common prefixes out of an alternation.
        assert_eq!(
            t("[A-Z]foo|[A-Z]quux"),
            hir_cat(vec![
                hir_uclass(&[('A', 'Z')]),
                hir_alt(vec![hir_lit("foo"), hir_lit("quux")]),
            ]),
        );
        assert_eq!(
            t("[A-Z][A-Z]|[A-Z]quux"),
            hir_cat(vec![
                hir_uclass(&[('A', 'Z')]),
                hir_alt(vec![hir_uclass(&[('A', 'Z')]), hir_lit("quux")]),
            ]),
        );
        assert_eq!(
            t("[A-Z][A-Z]|[A-Z][A-Z]quux"),
            hir_cat(vec![
                hir_uclass(&[('A', 'Z')]),
                hir_uclass(&[('A', 'Z')]),
                hir_alt(vec![Hir::empty(), hir_lit("quux")]),
            ]),
        );
        assert_eq!(
            t("[A-Z]foo|[A-Z]foobar"),
            hir_cat(vec![
                hir_uclass(&[('A', 'Z')]),
                hir_alt(vec![hir_lit("foo"), hir_lit("foobar")]),
            ]),
        );
    }

    #[test]
    fn regression_alt_empty_concat() {
        use crate::ast::{self, Ast};

        let span = Span::splat(Position::new(0, 0, 0));
        let ast = Ast::alternation(ast::Alternation {
            span,
            asts: vec![Ast::concat(ast::Concat { span, asts: vec![] })],
        });

        let mut t = Translator::new();
        assert_eq!(Ok(Hir::empty()), t.translate("", &ast));
    }

    #[test]
    fn regression_empty_alt() {
        use crate::ast::{self, Ast};

        let span = Span::splat(Position::new(0, 0, 0));
        let ast = Ast::concat(ast::Concat {
            span,
            asts: vec![Ast::alternation(ast::Alternation {
                span,
                asts: vec![],
            })],
        });

        let mut t = Translator::new();
        assert_eq!(Ok(Hir::fail()), t.translate("", &ast));
    }

    #[test]
    fn regression_singleton_alt() {
        use crate::{
            ast::{self, Ast},
            hir::Dot,
        };

        let span = Span::splat(Position::new(0, 0, 0));
        let ast = Ast::concat(ast::Concat {
            span,
            asts: vec![Ast::alternation(ast::Alternation {
                span,
                asts: vec![Ast::dot(span)],
            })],
        });

        let mut t = Translator::new();
        assert_eq!(Ok(Hir::dot(Dot::AnyCharExceptLF)), t.translate("", &ast));
    }

    // See: https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=63168
    #[test]
    fn regression_fuzz_match() {
        let pat = "[(\u{6} \0-\u{afdf5}]  \0 ";
        let ast = ParserBuilder::new()
            .octal(false)
            .ignore_whitespace(true)
            .build()
            .parse(pat)
            .unwrap();
        let hir = TranslatorBuilder::new()
            .utf8(true)
            .case_insensitive(false)
            .multi_line(false)
            .dot_matches_new_line(false)
            .swap_greed(true)
            .unicode(true)
            .build()
            .translate(pat, &ast)
            .unwrap();
        assert_eq!(
            hir,
            Hir::concat(vec![
                hir_uclass(&[('\0', '\u{afdf5}')]),
                hir_lit("\0"),
            ])
        );
    }

    // See: https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=63155
    #[cfg(feature = "unicode")]
    #[test]
    fn regression_fuzz_difference1() {
        let pat = r"\W\W|\W[^\v--\W\W\P{Script_Extensions:Pau_Cin_Hau}\u10A1A1-\U{3E3E3}--~~~~--~~~~~~~~------~~~~~~--~~~~~~]*";
        let _ = t(pat); // shouldn't panic
    }

    // See: https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=63153
    #[test]
    fn regression_fuzz_char_decrement1() {
        let pat = "w[w[^w?\rw\rw[^w?\rw[^w?\rw[^w?\rw[^w?\rw[^w?\rw[^w?\r\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0w?\rw[^w?\rw[^w?\rw[^w\0\0\u{1}\0]\0\0-*\0]\0\0\0\0\0\0\u{1}\0]\0\0-*\0]\0\0\0\0\0\u{1}\0]\0\0\0\0\0\0\0\0\0*\0\0\u{1}\0]\0\0-*\0][^w?\rw[^w?\rw[^w?\rw[^w?\rw[^w?\rw[^w?\rw[^w\0\0\u{1}\0]\0\0-*\0]\0\0\0\0\0\0\u{1}\0]\0\0-*\0]\0\0\0\0\0\u{1}\0]\0\0\0\0\0\0\0\0\0x\0\0\u{1}\0]\0\0-*\0]\0\0\0\0\0\0\0\0\0*??\0\u{7f}{2}\u{10}??\0\0\0\0\0\0\0\0\0\u{3}\0\0\0}\0-*\0]\0\0\0\0\0\0\u{1}\0]\0\0-*\0]\0\0\0\0\0\0\u{1}\0]\0\0-*\0]\0\0\0\0\0\u{1}\0]\0\0-*\0]\0\0\0\0\0\0\0\u{1}\0]\0\u{1}\u{1}H-i]-]\0\0\0\0\u{1}\0]\0\0\0\u{1}\0]\0\0-*\0\0\0\0\u{1}9-\u{7f}]\0'|-\u{7f}]\0'|(?i-ux)[-\u{7f}]\0'\u{3}\0\0\0}\0-*\0]<D\0\0\0\0\0\0\u{1}]\0\0\0\0]\0\0-*\0]\0\0 ";
        let _ = t(pat); // shouldn't panic
    }
}
