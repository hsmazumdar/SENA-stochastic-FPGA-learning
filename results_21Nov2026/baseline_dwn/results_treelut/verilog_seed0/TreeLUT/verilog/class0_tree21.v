module class0_tree21(input wire [467:0] i, output wire [0:0] o);

reg [0:0] o_reg;
assign o = o_reg;

always @(*) begin
	if(((~i[405]) & i[232]) | (i[62] & (~i[232])) | ((~i[62]) & (~i[232])) | ((~i[112]) & i[129] & i[405] & i[232]) | (i[377] & (~i[129]) & i[405] & i[232]) | (i[378] & i[112] & i[129] & i[405] & i[232]) | ((~i[378]) & i[112] & i[129] & i[405] & i[232]) | ((~i[43]) & (~i[377]) & (~i[129]) & i[405] & i[232]))
		o_reg = 0;
	else
		o_reg = 1;
end

endmodule
