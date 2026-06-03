# Reproducible PDF builds.
#
# pdfTeX stamps each PDF with /CreationDate, /ModDate, and an /ID seeded from the
# wall clock, so an UNCHANGED document still recompiles to different bytes - churn
# we do not want on a git-tracked main.pdf. Pinning SOURCE_DATE_EPOCH fixes those
# fields, so main.pdf is byte-identical across rebuilds and its bytes change only
# when the rendered content actually changes.
#
# The value is a deliberate fixed sentinel (2020-01-01 UTC), not a real build
# time; the date lives only in PDF metadata, never in the rendered document.
$ENV{'SOURCE_DATE_EPOCH'} = '1577836800';
$ENV{'FORCE_SOURCE_DATE'} = '1';
