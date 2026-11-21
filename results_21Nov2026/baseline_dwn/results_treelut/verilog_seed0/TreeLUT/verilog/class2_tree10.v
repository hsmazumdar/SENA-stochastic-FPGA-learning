module class2_tree10(input wire [467:0] i, output wire [0:0] o);

reg [0:0] o_reg;
assign o = o_reg;

always @(*) begin
	if(((~i[138]) & (~i[350])) | ((~i[456]) & i[91] & i[350]) | (i[390] & (~i[91]) & i[350]) | ((~i[452]) & i[456] & i[91] & i[350]))
		o_reg = 0;
	else
		o_reg = 1;
end

endmodule
