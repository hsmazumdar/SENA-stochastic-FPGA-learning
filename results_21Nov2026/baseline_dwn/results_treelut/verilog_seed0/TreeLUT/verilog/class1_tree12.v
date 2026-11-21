module class1_tree12(input wire [467:0] i, output wire [0:0] o);

reg [0:0] o_reg;
assign o = o_reg;

always @(*) begin
	if((i[395] & (~i[182])) | ((~i[57]) & i[349] & i[182]) | ((~i[165]) & i[84] & i[57] & i[349] & i[182]) | (i[148] & (~i[84]) & i[57] & i[349] & i[182]))
		o_reg = 0;
	else
		o_reg = 1;
end

endmodule
