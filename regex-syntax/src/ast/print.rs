/*!
This module provides a regular expression printer for `Ast`.
*/

use core::fmt;

use alloc::{format, string::{String, ToString}};

use recursion::{expand_and_collapse, PartiallyApplied};

use crate::ast::{self, visitor::{AstFrame, project_ast}, Ast};

/// A printer for a regular expression abstract syntax tree.
///
/// A printer converts an abstract syntax tree (AST) to a regular expression
/// pattern string. This particular printer uses constant stack space and heap
/// space proportional to the size of the AST.
///
/// This printer will not necessarily preserve the original formatting of the
/// regular expression pattern string. For example, all whitespace and comments
/// are ignored.
#[derive(Debug)]
pub struct Printer {
    _priv: (),
}

impl Printer {
    /// Create a new printer.
    pub fn new() -> Printer {
        Printer { _priv: () }
    }

    /// Print the given `Ast` to the given writer. The writer must implement
    /// `fmt::Write`. Typical implementations of `fmt::Write` that can be used
    /// here are a `fmt::Formatter` (which is available in `fmt::Display`
    /// implementations) or a `&mut String`.
    pub fn print<W: fmt::Write>(&mut self, ast: &Ast, mut wtr: W) -> fmt::Result {
        let result = print_ast(ast);
        wtr.write_str(&result)
    }
}

/// Print an AST to a string using the recursion crate's stack-safe traversal.
fn print_ast(ast: &Ast) -> String {
    expand_and_collapse::<AstFrame<PartiallyApplied>, _, _>(
        ast,
        |node| project_ast(node),
        |frame| match frame {
            AstFrame::Empty(_) => String::new(),
            AstFrame::Flags(x) => fmt_set_flags(&x),
            AstFrame::Literal(x) => fmt_literal(&x),
            AstFrame::Dot(_) => ".".to_string(),
            AstFrame::Assertion(x) => fmt_assertion(&x),
            AstFrame::ClassPerl(x) => fmt_class_perl(&x),
            AstFrame::ClassUnicode(x) => fmt_class_unicode(&x),
            AstFrame::ClassBracketed(x) => fmt_class_bracketed(&x),
            AstFrame::Repetition { op, greedy, child, .. } => {
                let mut s: String = child;
                s.push_str(&fmt_repetition_op(&op, greedy));
                s
            }
            AstFrame::Group { kind, child, .. } => {
                let mut s = fmt_group_open(&kind);
                s.push_str(&child);
                s.push(')');
                s
            }
            AstFrame::Concat { children, .. } => {
                children.into_iter().collect()
            }
            AstFrame::Alternation { children, .. } => {
                children.join("|")
            }
        },
    )
}

fn fmt_group_open(kind: &ast::GroupKind) -> String {
    use ast::GroupKind::*;
    match kind {
        CaptureIndex(_) => "(".to_string(),
        CaptureName { ref name, starts_with_p } => {
            if *starts_with_p {
                format!("(?P<{}>", name.name)
            } else {
                format!("(?<{}>", name.name)
            }
        }
        NonCapturing(ref flags) => {
            format!("(?{}:", fmt_flags(flags))
        }
    }
}

fn fmt_repetition_op(op: &ast::RepetitionOp, greedy: bool) -> String {
    use ast::RepetitionKind::*;
    match op.kind {
        ZeroOrOne if greedy => "?".to_string(),
        ZeroOrOne => "??".to_string(),
        ZeroOrMore if greedy => "*".to_string(),
        ZeroOrMore => "*?".to_string(),
        OneOrMore if greedy => "+".to_string(),
        OneOrMore => "+?".to_string(),
        Range(ref x) => {
            let base = fmt_repetition_range(x);
            if greedy { base } else { format!("{}?", base) }
        }
    }
}

fn fmt_repetition_range(ast: &ast::RepetitionRange) -> String {
    use ast::RepetitionRange::*;
    match *ast {
        Exactly(x) => format!("{{{}}}", x),
        AtLeast(x) => format!("{{{},}}", x),
        Bounded(x, y) => format!("{{{},{}}}", x, y),
    }
}

fn fmt_literal(ast: &ast::Literal) -> String {
    use ast::LiteralKind::*;

    match ast.kind {
        Verbatim => ast.c.to_string(),
        Meta | Superfluous => format!(r"\{}", ast.c),
        Octal => format!(r"\{:o}", u32::from(ast.c)),
        HexFixed(ast::HexLiteralKind::X) => {
            format!(r"\x{:02X}", u32::from(ast.c))
        }
        HexFixed(ast::HexLiteralKind::UnicodeShort) => {
            format!(r"\u{:04X}", u32::from(ast.c))
        }
        HexFixed(ast::HexLiteralKind::UnicodeLong) => {
            format!(r"\U{:08X}", u32::from(ast.c))
        }
        HexBrace(ast::HexLiteralKind::X) => {
            format!(r"\x{{{:X}}}", u32::from(ast.c))
        }
        HexBrace(ast::HexLiteralKind::UnicodeShort) => {
            format!(r"\u{{{:X}}}", u32::from(ast.c))
        }
        HexBrace(ast::HexLiteralKind::UnicodeLong) => {
            format!(r"\U{{{:X}}}", u32::from(ast.c))
        }
        Special(ast::SpecialLiteralKind::Bell) => r"\a".to_string(),
        Special(ast::SpecialLiteralKind::FormFeed) => r"\f".to_string(),
        Special(ast::SpecialLiteralKind::Tab) => r"\t".to_string(),
        Special(ast::SpecialLiteralKind::LineFeed) => r"\n".to_string(),
        Special(ast::SpecialLiteralKind::CarriageReturn) => r"\r".to_string(),
        Special(ast::SpecialLiteralKind::VerticalTab) => r"\v".to_string(),
        Special(ast::SpecialLiteralKind::Space) => r"\ ".to_string(),
    }
}

fn fmt_assertion(ast: &ast::Assertion) -> String {
    use ast::AssertionKind::*;
    match ast.kind {
        StartLine => "^".to_string(),
        EndLine => "$".to_string(),
        StartText => r"\A".to_string(),
        EndText => r"\z".to_string(),
        WordBoundary => r"\b".to_string(),
        NotWordBoundary => r"\B".to_string(),
        WordBoundaryStart => r"\b{start}".to_string(),
        WordBoundaryEnd => r"\b{end}".to_string(),
        WordBoundaryStartAngle => r"\<".to_string(),
        WordBoundaryEndAngle => r"\>".to_string(),
        WordBoundaryStartHalf => r"\b{start-half}".to_string(),
        WordBoundaryEndHalf => r"\b{end-half}".to_string(),
    }
}

fn fmt_set_flags(ast: &ast::SetFlags) -> String {
    format!("(?{})", fmt_flags(&ast.flags))
}

fn fmt_flags(ast: &ast::Flags) -> String {
    use ast::{Flag, FlagsItemKind};

    let mut s = String::new();
    for item in &ast.items {
        match item.kind {
            FlagsItemKind::Negation => s.push('-'),
            FlagsItemKind::Flag(ref flag) => match *flag {
                Flag::CaseInsensitive => s.push('i'),
                Flag::MultiLine => s.push('m'),
                Flag::DotMatchesNewLine => s.push('s'),
                Flag::SwapGreed => s.push('U'),
                Flag::Unicode => s.push('u'),
                Flag::CRLF => s.push('R'),
                Flag::IgnoreWhitespace => s.push('x'),
            },
        }
    }
    s
}

fn fmt_class_bracketed(ast: &ast::ClassBracketed) -> String {
    let mut s = if ast.negated {
        "[^".to_string()
    } else {
        "[".to_string()
    };
    s.push_str(&fmt_class_set(&ast.kind));
    s.push(']');
    s
}

fn fmt_class_set(set: &ast::ClassSet) -> String {
    match set {
        ast::ClassSet::Item(item) => fmt_class_set_item(item),
        ast::ClassSet::BinaryOp(op) => fmt_class_set_binary_op(op),
    }
}

fn fmt_class_set_item(item: &ast::ClassSetItem) -> String {
    use ast::ClassSetItem::*;
    match item {
        Empty(_) => String::new(),
        Literal(x) => fmt_literal(x),
        Range(x) => {
            format!("{}-{}", fmt_literal(&x.start), fmt_literal(&x.end))
        }
        Ascii(x) => fmt_class_ascii(x),
        Unicode(x) => fmt_class_unicode(x),
        Perl(x) => fmt_class_perl(x),
        Bracketed(x) => fmt_class_bracketed(x),
        Union(x) => {
            x.items.iter().map(fmt_class_set_item).collect()
        }
    }
}

fn fmt_class_set_binary_op(op: &ast::ClassSetBinaryOp) -> String {
    let lhs = fmt_class_set(&op.lhs);
    let rhs = fmt_class_set(&op.rhs);
    let op_str = match op.kind {
        ast::ClassSetBinaryOpKind::Intersection => "&&",
        ast::ClassSetBinaryOpKind::Difference => "--",
        ast::ClassSetBinaryOpKind::SymmetricDifference => "~~",
    };
    format!("{}{}{}", lhs, op_str, rhs)
}

fn fmt_class_perl(ast: &ast::ClassPerl) -> String {
    use ast::ClassPerlKind::*;
    match ast.kind {
        Digit if ast.negated => r"\D".to_string(),
        Digit => r"\d".to_string(),
        Space if ast.negated => r"\S".to_string(),
        Space => r"\s".to_string(),
        Word if ast.negated => r"\W".to_string(),
        Word => r"\w".to_string(),
    }
}

fn fmt_class_ascii(ast: &ast::ClassAscii) -> String {
    use ast::ClassAsciiKind::*;
    match ast.kind {
        Alnum if ast.negated => "[:^alnum:]".to_string(),
        Alnum => "[:alnum:]".to_string(),
        Alpha if ast.negated => "[:^alpha:]".to_string(),
        Alpha => "[:alpha:]".to_string(),
        Ascii if ast.negated => "[:^ascii:]".to_string(),
        Ascii => "[:ascii:]".to_string(),
        Blank if ast.negated => "[:^blank:]".to_string(),
        Blank => "[:blank:]".to_string(),
        Cntrl if ast.negated => "[:^cntrl:]".to_string(),
        Cntrl => "[:cntrl:]".to_string(),
        Digit if ast.negated => "[:^digit:]".to_string(),
        Digit => "[:digit:]".to_string(),
        Graph if ast.negated => "[:^graph:]".to_string(),
        Graph => "[:graph:]".to_string(),
        Lower if ast.negated => "[:^lower:]".to_string(),
        Lower => "[:lower:]".to_string(),
        Print if ast.negated => "[:^print:]".to_string(),
        Print => "[:print:]".to_string(),
        Punct if ast.negated => "[:^punct:]".to_string(),
        Punct => "[:punct:]".to_string(),
        Space if ast.negated => "[:^space:]".to_string(),
        Space => "[:space:]".to_string(),
        Upper if ast.negated => "[:^upper:]".to_string(),
        Upper => "[:upper:]".to_string(),
        Word if ast.negated => "[:^word:]".to_string(),
        Word => "[:word:]".to_string(),
        Xdigit if ast.negated => "[:^xdigit:]".to_string(),
        Xdigit => "[:xdigit:]".to_string(),
    }
}

fn fmt_class_unicode(ast: &ast::ClassUnicode) -> String {
    use ast::ClassUnicodeKind::*;
    use ast::ClassUnicodeOpKind::*;

    let prefix = if ast.negated { r"\P" } else { r"\p" };
    match &ast.kind {
        OneLetter(c) => format!("{}{}", prefix, c),
        Named(x) => format!("{}{{{}}}", prefix, x),
        NamedValue { op: Equal, name, value } => {
            format!("{}{{{}={}}}", prefix, name, value)
        }
        NamedValue { op: Colon, name, value } => {
            format!("{}{{{}:{}}}", prefix, name, value)
        }
        NamedValue { op: NotEqual, name, value } => {
            format!("{}{{{}!={}}}", prefix, name, value)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ast::parse::ParserBuilder;

    use super::*;

    fn roundtrip(given: &str) {
        roundtrip_with(|b| b, given);
    }

    fn roundtrip_with<F>(mut f: F, given: &str)
    where
        F: FnMut(&mut ParserBuilder) -> &mut ParserBuilder,
    {
        let mut builder = ParserBuilder::new();
        f(&mut builder);
        let ast = builder.build().parse(given).unwrap();

        let mut printer = Printer::new();
        let mut dst = String::new();
        printer.print(&ast, &mut dst).unwrap();
        assert_eq!(given, dst);
    }

    #[test]
    fn print_literal() {
        roundtrip("a");
        roundtrip(r"\[");
        roundtrip_with(|b| b.octal(true), r"\141");
        roundtrip(r"\x61");
        roundtrip(r"\x7F");
        roundtrip(r"\u0061");
        roundtrip(r"\U00000061");
        roundtrip(r"\x{61}");
        roundtrip(r"\x{7F}");
        roundtrip(r"\u{61}");
        roundtrip(r"\U{61}");

        roundtrip(r"\a");
        roundtrip(r"\f");
        roundtrip(r"\t");
        roundtrip(r"\n");
        roundtrip(r"\r");
        roundtrip(r"\v");
        roundtrip(r"(?x)\ ");
    }

    #[test]
    fn print_dot() {
        roundtrip(".");
    }

    #[test]
    fn print_concat() {
        roundtrip("ab");
        roundtrip("abcde");
        roundtrip("a(bcd)ef");
    }

    #[test]
    fn print_alternation() {
        roundtrip("a|b");
        roundtrip("a|b|c|d|e");
        roundtrip("|a|b|c|d|e");
        roundtrip("|a|b|c|d|e|");
        roundtrip("a(b|c|d)|e|f");
    }

    #[test]
    fn print_assertion() {
        roundtrip(r"^");
        roundtrip(r"$");
        roundtrip(r"\A");
        roundtrip(r"\z");
        roundtrip(r"\b");
        roundtrip(r"\B");
    }

    #[test]
    fn print_repetition() {
        roundtrip("a?");
        roundtrip("a??");
        roundtrip("a*");
        roundtrip("a*?");
        roundtrip("a+");
        roundtrip("a+?");
        roundtrip("a{5}");
        roundtrip("a{5}?");
        roundtrip("a{5,}");
        roundtrip("a{5,}?");
        roundtrip("a{5,10}");
        roundtrip("a{5,10}?");
    }

    #[test]
    fn print_flags() {
        roundtrip("(?i)");
        roundtrip("(?-i)");
        roundtrip("(?s-i)");
        roundtrip("(?-si)");
        roundtrip("(?siUmux)");
    }

    #[test]
    fn print_group() {
        roundtrip("(?i:a)");
        roundtrip("(?P<foo>a)");
        roundtrip("(?<foo>a)");
        roundtrip("(a)");
    }

    #[test]
    fn print_class() {
        roundtrip(r"[abc]");
        roundtrip(r"[a-z]");
        roundtrip(r"[^a-z]");
        roundtrip(r"[a-z0-9]");
        roundtrip(r"[-a-z0-9]");
        roundtrip(r"[-a-z0-9]");
        roundtrip(r"[a-z0-9---]");
        roundtrip(r"[a-z&&m-n]");
        roundtrip(r"[[a-z&&m-n]]");
        roundtrip(r"[a-z--m-n]");
        roundtrip(r"[a-z~~m-n]");
        roundtrip(r"[a-z[0-9]]");
        roundtrip(r"[a-z[^0-9]]");

        roundtrip(r"\d");
        roundtrip(r"\D");
        roundtrip(r"\s");
        roundtrip(r"\S");
        roundtrip(r"\w");
        roundtrip(r"\W");

        roundtrip(r"[[:alnum:]]");
        roundtrip(r"[[:^alnum:]]");
        roundtrip(r"[[:alpha:]]");
        roundtrip(r"[[:^alpha:]]");
        roundtrip(r"[[:ascii:]]");
        roundtrip(r"[[:^ascii:]]");
        roundtrip(r"[[:blank:]]");
        roundtrip(r"[[:^blank:]]");
        roundtrip(r"[[:cntrl:]]");
        roundtrip(r"[[:^cntrl:]]");
        roundtrip(r"[[:digit:]]");
        roundtrip(r"[[:^digit:]]");
        roundtrip(r"[[:graph:]]");
        roundtrip(r"[[:^graph:]]");
        roundtrip(r"[[:lower:]]");
        roundtrip(r"[[:^lower:]]");
        roundtrip(r"[[:print:]]");
        roundtrip(r"[[:^print:]]");
        roundtrip(r"[[:punct:]]");
        roundtrip(r"[[:^punct:]]");
        roundtrip(r"[[:space:]]");
        roundtrip(r"[[:^space:]]");
        roundtrip(r"[[:upper:]]");
        roundtrip(r"[[:^upper:]]");
        roundtrip(r"[[:word:]]");
        roundtrip(r"[[:^word:]]");
        roundtrip(r"[[:xdigit:]]");
        roundtrip(r"[[:^xdigit:]]");

        roundtrip(r"\pL");
        roundtrip(r"\PL");
        roundtrip(r"\p{L}");
        roundtrip(r"\P{L}");
        roundtrip(r"\p{X=Y}");
        roundtrip(r"\P{X=Y}");
        roundtrip(r"\p{X:Y}");
        roundtrip(r"\P{X:Y}");
        roundtrip(r"\p{X!=Y}");
        roundtrip(r"\P{X!=Y}");
    }
}
