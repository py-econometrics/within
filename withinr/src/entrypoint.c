// Forward routine registration from C to the Rust static library.
// R's dynamic loader calls R_init_withinr(); extendr generates
// R_init_withinr_extendr() inside the Rust code.

void R_init_withinr_extendr(void *dll);

void R_init_withinr(void *dll) {
    R_init_withinr_extendr(dll);
}
