module class0_tree19(input wire [467:0] i, output wire [0:0] o);

reg [0:0] o_reg;
assign o = o_reg;

always @(*) begin
	if(((~i[39]) & i[233]) | (i[149] & (~i[233])) | (i[185] & (~i[149]) & (~i[233])) | ((~i[185]) & (~i[149]) & (~i[233])) | ((~i[190]) & i[296] & i[39] & i[233]) | (i[167] & (~i[296]) & i[39] & i[233]) | ((~i[167]) & (~i[296]) & i[39] & i[233]) | ((~i[51]) & i[190] & i[296] & i[39] & i[233]))
		o_reg = 0;
	else
		o_reg = 1;
end

endmodule
