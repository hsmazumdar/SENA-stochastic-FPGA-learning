module class7_tree10(input wire [467:0] i, output wire [0:0] o);

reg [0:0] o_reg;
assign o = o_reg;

always @(*) begin
	if(((~i[49])) | (i[137] & (~i[197]) & i[49]) | (i[149] & i[79] & i[383] & i[197] & i[49]) | ((~i[166]) & (~i[79]) & i[383] & i[197] & i[49]) | ((~i[232]) & i[456] & (~i[383]) & i[197] & i[49]) | ((~i[62]) & (~i[456]) & (~i[383]) & i[197] & i[49]))
		o_reg = 0;
	else
		o_reg = 1;
end

endmodule
